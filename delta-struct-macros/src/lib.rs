//! Procedural macro implementation behind the `delta-struct` crate.
//!
//! Use [`delta-struct`](https://docs.rs/delta-struct) rather than depending on
//! this crate directly; it re-exports the [`Delta`] derive alongside the trait
//! the derive generates an implementation of, and carries the user-facing
//! documentation.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenTree};
use proc_macro_error::{abort_call_site, proc_macro_error};
use quote::{format_ident, quote};
use std::{iter::FromIterator, str::FromStr};
use syn::{
    parse_macro_input, punctuated::Punctuated, Attribute, Data, DeriveInput, Fields, Ident, Lit,
    Meta, MetaList, MetaNameValue, NestedMeta, Path, PredicateType, Token, TraitBound,
    TraitBoundModifier, Type, TypeParamBound, WherePredicate,
};

/// How a single field is diffed, and therefore how it is represented on the
/// generated delta struct.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum FieldType {
    /// A positional diff: the delta is a Myers edit script over the sequence.
    Ordered,
    /// A bag of items: the delta records additions and removals, not order.
    Unordered,
    /// A bag of key/value entries: like [`FieldType::Unordered`], except that
    /// entries sharing a key are diffed with the value's own `Delta` rather
    /// than recorded as a removal plus an addition.
    UnorderedDelta,
    /// Compared with `!=` and replaced wholesale.
    Scalar,
    /// Diffed recursively via the field type's own `Delta` implementation.
    Delta,
}

const VALID_FIELD_TYPES: &str =
    "\"ordered\", \"unordered\", \"unordered-delta\", \"delta\", or \"scalar\"";

/// One field of the source struct, as the code generators want it: its name
/// (or, for a tuple struct, its index), its declared type, how it is diffed,
/// and the tokens to emit above the field it turns into.
type Field = (String, Type, FieldType, String);

/// One field as it comes back from attribute parsing, before the container's
/// `default` has been used to fill in a missing `field_type`.
type ParsedField = (String, Type, ParsedAttrs);

/// The `(field type, delta_leader)` pair a single `#[delta_struct(...)]`
/// yields, or the reason it could not be read.
type ParsedAttrs = Result<(Option<FieldType>, String), FieldTypeError>;

/// Derives `Delta`, generating a `{Self}Delta` struct that holds only the
/// changed parts of a value plus the trait implementation that produces and
/// applies one.
///
/// The generated struct takes the visibility and generic parameters of the
/// type it is derived on, and all of its fields are `pub`. A tuple struct's
/// delta is a tuple struct in turn, with its fields in the same positions.
///
/// See the [`delta-struct`](https://docs.rs/delta-struct) crate documentation
/// for the full picture, including trait bounds, serde usage, and limitations;
/// what follows is the attribute reference.
///
/// # Container attributes
///
/// | Attribute | Effect |
/// | --- | --- |
/// | `default = "<field type>"` | Field type for fields that don't specify one. Defaults to `"scalar"`. |
/// | `delta_leader = "<tokens>"` | Tokens emitted directly above the generated struct — derives, doc comments, anything. |
///
/// # Field attributes
///
/// | Attribute | Effect |
/// | --- | --- |
/// | `field_type = "<field type>"` | How this field is diffed. Overrides the container's `default`. |
/// | `delta_leader = "<tokens>"` | Tokens emitted directly above the generated field. |
///
/// # Field types
///
/// Each maps one source field onto exactly one delta field.
///
/// | Value | Delta representation | Requires |
/// | --- | --- | --- |
/// | `"scalar"` | `Option<T>` | `T: PartialEq` |
/// | `"unordered"` | `BagDelta<Item>`, an `add` and a `remove` | `T: IntoIterator + Extend<Item> + TryIndex<Item, Output = Item>` |
/// | `"unordered-delta"` | `MapDelta<Key, Value, <Value as Delta>::Output>`, an `add`, a `remove`, and a `change` | `T: IntoIterator + Extend<Item> + TryIndexMut<Key, Output = Value> Item: MapEntry` (so `(K, V)`), `Value: Delta` |
/// | `"ordered"` | `SeqDelta<Item>`, a Myers edit script | `T: IntoIterator + FromIterator<Item>`, `Item: Hash + Eq` |
/// | `"delta"` | `Option<<T as Delta>::Output>` | `T: Delta` |
///
/// # Example
///
/// ```ignore
/// use delta_struct::Delta;
///
/// #[derive(Delta)]
/// #[delta_struct(delta_leader = "#[derive(Debug)]")]
/// struct Device {
///     #[delta_struct(field_type = "unordered")]
///     services: std::collections::HashSet<String>,
///     online: bool,
/// }
/// ```
#[proc_macro_derive(Delta, attributes(delta_struct))]
#[proc_macro_error]
pub fn derive_delta(input: TokenStream) -> TokenStream {
    let DeriveInput {
        attrs,
        vis,
        ident,
        mut generics,
        data,
    } = parse_macro_input!(input as DeriveInput);
    let (default_field_type, delta_leader) =
        match get_fieldtype_from_attrs(attrs.into_iter(), "default") {
            Ok((v, delta_leader)) => (v.unwrap_or(FieldType::Scalar), delta_leader),
            Err(_) => {
                abort_call_site!(
                    "delta_struct(default = ...) for {} is not an accepted value, expected {}.",
                    ident,
                    VALID_FIELD_TYPES
                );
            }
        };

    let (named, fields) = match data {
        Data::Struct(strukt) => match strukt.fields {
            Fields::Named(named) => (
                true,
                collect_results(
                    named.named.into_iter().map(|field| {
                        (
                            field.ident.unwrap().to_string(),
                            field.ty,
                            get_fieldtype_from_attrs(field.attrs.into_iter(), "field_type"),
                        )
                    }),
                    default_field_type,
                ),
            ),
            Fields::Unnamed(unnamed) => (
                false,
                collect_results(
                    unnamed.unnamed.into_iter().enumerate().map(|(i, field)| {
                        (
                            i.to_string(),
                            field.ty,
                            get_fieldtype_from_attrs(field.attrs.into_iter(), "field_type"),
                        )
                    }),
                    default_field_type,
                ),
            ),
            Fields::Unit => (false, Ok(vec![])),
        },
        _ => {
            abort_call_site!(
                "delta_struct::Delta may only be derived for struct types currently. {} is not a struct type."
            , ident)
        }
    };
    let fields = match fields {
        Ok(fields) => fields,
        Err(bad_fields) => {
            let bad_fields = format!("{:?}", bad_fields);
            abort_call_site!(
                "delta_struct(field_type = ...) for fields in {}: {} are not valid values. Expected {}.",
                ident,
                bad_fields,
                VALID_FIELD_TYPES
            )
        }
    };
    let delta_leader = match proc_macro2::TokenStream::from_str(&delta_leader) {
        Ok(v) => v,
        Err(e) => {
            abort_call_site!("error parsing delta leader as token stream {}", e);
        }
    };
    let delta_ident = format_ident!("{}Delta", ident);
    let delta_fields = delta_fields(named, fields.iter().cloned());
    // The delta struct repeats the source type's generics verbatim, bounds and
    // all, since its fields can project through them — `<T as Delta>::Output`
    // for a delta field, `<T as IntoIterator>::Item` for an unordered one. Grab
    // the where clause before the `PartialEq` predicates below are pushed onto
    // it; those are the impl's business, not the struct's.
    let og_where_clause = generics.where_clause.clone();
    let (delta_compute_let, delta_compute_fields) =
        delta_compute_fields(named, fields.iter().cloned());
    let (delta_apply_let, delta_apply_actions) = delta_apply_fields(named, fields.into_iter());
    // A tuple struct's delta is a tuple struct too, which means the
    // declaration, the initializer, and the destructuring pattern all have to
    // switch from braces to parentheses together. Two things differ beyond the
    // brackets: a tuple struct puts its `where` clause *after* the fields and
    // ends in a semicolon, and its constructor lives in the value namespace,
    // which `Self::Output` — an associated type — cannot reach, so the
    // initializer and pattern name the struct itself and let inference supply
    // its generics.
    let (delta_struct, delta_compute_init, delta_apply_pattern) = if named {
        (
            quote! {
                #delta_leader
                #vis struct #delta_ident #generics #og_where_clause {
                    #delta_fields
                }
            },
            quote!(Self::Output { #delta_compute_fields }),
            quote!(Self::Output { #delta_apply_let }),
        )
    } else {
        (
            quote! {
                #delta_leader
                #vis struct #delta_ident #generics (#delta_fields) #og_where_clause;
            },
            quote!(#delta_ident(#delta_compute_fields)),
            quote!(#delta_ident(#delta_apply_let)),
        )
    };
    // Scalar and unordered fields compare values with `==`, so every type
    // parameter picks up a `PartialEq` bound on the impl. This is broader than
    // strictly necessary — a parameter used only by a `delta` field does not
    // need it.
    let partial_eq_types = generics
        .type_params()
        .map(|t| t.ident.clone())
        .collect::<Vec<_>>();
    let where_clause = generics.make_where_clause();
    for ty in partial_eq_types {
        let mut bounds = Punctuated::new();
        let mut segments = Punctuated::new();
        segments.push(Ident::new("std", Span::call_site()).into());
        segments.push(Ident::new("cmp", Span::call_site()).into());
        segments.push(Ident::new("PartialEq", Span::call_site()).into());
        bounds.push(TypeParamBound::Trait(TraitBound {
            paren_token: None,
            modifier: TraitBoundModifier::None,
            lifetimes: None,
            path: Path {
                leading_colon: Some(Token!(::)(Span::call_site())),
                segments,
            },
        }));
        where_clause
            .predicates
            .push(WherePredicate::Type(PredicateType {
                lifetimes: None,
                bounded_ty: Type::Verbatim(<Ident as Into<TokenTree>>::into(ty).into()),
                colon_token: Token!(:)(Span::call_site()),
                bounds,
            }));
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let delta_impl = quote! {
      impl #impl_generics Delta for #ident #ty_generics #where_clause  {
          // `ty_generics` and not `generics`: the latter renders parameter
          // bounds too, which are not allowed in a type position.
          type Output = #delta_ident #ty_generics;

          fn delta(old: Self, new: Self) -> Option<Self::Output> {
           let mut delta_is_some = false;
           #delta_compute_let
           if delta_is_some {
               Some(#delta_compute_init)
           } else {
               None
           }
          }

          fn apply_delta(&mut self, delta: Self::Output) {
            let #delta_apply_pattern = delta;
            #delta_apply_actions
          }
      }
    };
    let output = quote! {
        #delta_struct

        #delta_impl
    };
    TokenStream::from(output)
}

/// Emits the field declarations of the generated delta struct.
///
/// Fields arrive as `(name, type, field type, delta_leader)`, where `name` is
/// the source field's name or, for tuple structs, its index. `named` says
/// which of the two it is, and so whether these declarations are about to be
/// wrapped in braces or in parentheses: a tuple struct's delta is a tuple
/// struct too, and its fields are positional rather than named.
fn delta_fields(named: bool, iter: impl Iterator<Item = Field>) -> proc_macro2::TokenStream {
    FromIterator::from_iter(iter.map(|(ident, ty, field_ty, field_leader)| {
        let field_leader = proc_macro2::TokenStream::from_str(&field_leader).unwrap();
        let declared_ty = match field_ty {
            FieldType::Ordered => {
                quote!(::delta_struct::SeqDelta<<#ty as ::std::iter::IntoIterator>::Item>)
            }
            FieldType::Unordered => {
                quote!(::delta_struct::BagDelta<<#ty as ::std::iter::IntoIterator>::Item>)
            }
            FieldType::UnorderedDelta => {
                // The field's own type names the collection, not its key and
                // value; `MapEntry` is what projects those back out of the
                // item type so the delta field can be spelled at all.
                let entry = quote!(<#ty as ::std::iter::IntoIterator>::Item);
                let key = quote!(<#entry as ::delta_struct::MapEntry>::Key);
                let value = quote!(<#entry as ::delta_struct::MapEntry>::Value);
                quote!(::delta_struct::MapDelta<#key, #value, <#value as Delta>::Output>)
            }
            FieldType::Scalar => quote!(::std::option::Option<#ty>),
            FieldType::Delta => quote!(::std::option::Option<<#ty as Delta>::Output>),
        };
        if named {
            let ident = format_ident!("{}", ident);
            quote! {
                #field_leader
                pub #ident: #declared_ty,
            }
        } else {
            quote! {
                #field_leader
                pub #declared_ty,
            }
        }
    }))
}

/// Emits the body of `Delta::delta`, as `(statements, struct initializer)`.
///
/// The statements bind one local per generated field and set `delta_is_some`
/// whenever they find a real change; the initializer then moves those locals
/// into the delta struct. Fields arrive in the same shape as in
/// [`delta_fields`].
fn delta_compute_fields(
    named: bool,
    iter: impl Iterator<Item = Field>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    iter.map(|(og_ident, _ty, field_ty, _field_leader)| {
        let ident = if named {
            format_ident!("{}", og_ident)
        } else {
            format_ident!("field_{}", og_ident)
        };
        let og_ident: proc_macro2::TokenStream = FromStr::from_str(&og_ident).unwrap();
        let statements = match field_ty {
            FieldType::Ordered | FieldType::Unordered | FieldType::UnorderedDelta => {
                let module = collection_module(field_ty);
                quote! {
                    let #ident = ::delta_struct::#module::diff(old.#og_ident, new.#og_ident);
                    delta_is_some = delta_is_some || !#ident.is_empty();
                }
            }
            FieldType::Scalar => quote! {
                let #ident = if old.#og_ident != new.#og_ident {
                    delta_is_some = true;
                    Some(new.#og_ident)
                } else {
                    None
                };
            },
            FieldType::Delta => quote! {
                let #ident = Delta::delta(old.#og_ident, new.#og_ident);
                delta_is_some = delta_is_some || #ident.is_some();
            },
        };
        // The locals are listed in declaration order, so this reads as a field
        // shorthand inside braces and as a positional argument inside parens —
        // whichever bracket the caller wraps it in.
        (statements, quote!(#ident,))
    })
    .unzip()
}

/// Emits the body of `Delta::apply_delta`, as `(destructuring pattern,
/// statements)`.
///
/// The pattern takes the delta struct apart into locals and the statements
/// write each change back into `self`. Fields arrive in the same shape as in
/// [`delta_fields`].
fn delta_apply_fields(
    named: bool,
    iter: impl Iterator<Item = Field>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    iter.map(|(og_ident, _ty, field_ty, _field_leader)| {
        let ident = if named {
            format_ident!("{}", og_ident)
        } else {
            format_ident!("field_{}", og_ident)
        };
        let og_ident: proc_macro2::TokenStream = FromStr::from_str(&og_ident).unwrap();
        let statements = match field_ty {
            FieldType::Ordered | FieldType::Unordered | FieldType::UnorderedDelta => {
                let module = collection_module(field_ty);
                quote! {
                    ::delta_struct::#module::apply(&mut self.#og_ident, #ident);
                }
            }
            FieldType::Scalar => quote! {
                if let Some(v) = #ident {
                    self.#og_ident = v;
                }
            },
            FieldType::Delta => quote! {
                if let Some(v) = #ident {
                    self.#og_ident.apply_delta(v);
                }
            },
        };
        // Binds one local per field, in declaration order — see the matching
        // note in `delta_compute_fields` about braces versus parens.
        (quote!(#ident,), statements)
    })
    .unzip()
}

/// The runtime module backing a collection field type.
///
/// The three collection field types differ in what their delta looks like, but
/// not in how the derive drives one: each module pairs a `diff` and an `apply`
/// over a delta type that reports whether it is empty. Panics for the two
/// non-collection field types, which the callers never pass.
fn collection_module(field_ty: FieldType) -> Ident {
    match field_ty {
        FieldType::Ordered => format_ident!("seq"),
        FieldType::Unordered => format_ident!("bag"),
        FieldType::UnorderedDelta => format_ident!("map"),
        FieldType::Scalar | FieldType::Delta => {
            unreachable!("{:?} is not a collection field type", field_ty)
        }
    }
}

/// Resolves each field's parsed attributes against the container default,
/// collecting *every* bad field rather than stopping at the first, so one
/// compile reports them all.
#[allow(clippy::manual_try_fold)] // Collects errors too
fn collect_results(
    iter: impl Iterator<Item = ParsedField>,
    default_field_type: FieldType,
) -> Result<Vec<Field>, Vec<String>> {
    iter.fold(Ok(vec![]), |v, i| match (v, i) {
        (Ok(mut v), (ident, b, Ok((c, d)))) => {
            v.push((ident, b, c.unwrap_or(default_field_type), d));
            Ok(v)
        }
        (Ok(_), (ident, _, Err(_))) => Err(vec![ident]),
        (Err(mut v), (ident, _, Err(_))) => {
            v.push(ident);
            Err(v)
        }
        (v @ Err(_), _) => v,
    })
}

enum FieldTypeError {
    /// The `delta_struct(...)` attribute contained entries that were not
    /// `name = "value"` pairs.
    UnrecognizedJunkFound,
}

/// Reads a `#[delta_struct(...)]` attribute, returning
/// `(field type, delta_leader)`.
///
/// `attr_name` is the key naming the field type in this position — `"default"`
/// on a container, `"field_type"` on a field — because the two spellings mean
/// the same thing at different scopes. The field type is `None` when the
/// attribute is absent or names no field type, leaving the caller to fill in
/// the default; `delta_leader` is empty when unspecified.
#[allow(clippy::manual_try_fold)] // Collects errors too
fn get_fieldtype_from_attrs(iter: impl Iterator<Item = Attribute>, attr_name: &str) -> ParsedAttrs {
    for attr in iter {
        if let Ok(Meta::List(MetaList { path, nested, .. })) = attr.parse_meta() {
            let Path { segments, .. } = path;
            if segments
                .iter()
                .map(|p| &p.ident)
                .eq(["delta_struct"].iter().cloned())
            {
                let values: Result<Vec<_>, Vec<NestedMeta>> = nested
                    .iter()
                    .map(|nested_meta| match nested_meta {
                        NestedMeta::Meta(Meta::NameValue(MetaNameValue {
                            path,
                            lit: Lit::Str(s),
                            ..
                        })) => Ok((path.get_ident().map(|i| i.to_string()), s.value())),
                        e => Err(e),
                    })
                    .fold(Ok(vec![]), |v, i| match (v, i) {
                        (Ok(mut v), Ok(i)) => {
                            v.push(i);
                            Ok(v)
                        }
                        (Ok(_), Err(e)) => Err(vec![e.clone()]),
                        (Err(mut v), Err(e)) => {
                            v.push(e.clone());
                            Err(v)
                        }
                        (v @ Err(_), _) => v,
                    });
                return match values {
                    Ok(v) => {
                        let mut field_type = None;
                        let mut delta_leader = String::new();
                        for i in v {
                            match i.0.as_deref() {
                                Some("delta_leader") => {
                                    delta_leader = i.1;
                                }
                                a if Some(attr_name) == a => {
                                    field_type = string_to_fieldtype(&i.1);
                                }
                                a => {
                                    abort_call_site!("Unrecognized value {:?}", a);
                                }
                            }
                        }
                        Ok((field_type, delta_leader))
                    }
                    Err(_) => Err(FieldTypeError::UnrecognizedJunkFound),
                };
            }
        }
    }
    Ok((None, String::new()))
}

/// Maps the attribute spelling of a field type to its variant, or `None` if it
/// is not one of the recognized names.
fn string_to_fieldtype(s: &str) -> Option<FieldType> {
    match s {
        "ordered" => Some(FieldType::Ordered),
        "unordered" => Some(FieldType::Unordered),
        "unordered-delta" => Some(FieldType::UnorderedDelta),
        "scalar" => Some(FieldType::Scalar),
        "delta" => Some(FieldType::Delta),
        _ => None,
    }
}

/// Derives `Fingerprint`, a stable content hash used to check that a delta is
/// being applied to the state it was computed against.
///
/// Walks a struct's fields in declaration order, or an enum's variant index
/// followed by that variant's fields. Every field type has to implement
/// `Fingerprint` too, and every type parameter picks up a `Fingerprint` bound.
///
/// Unlike the `Delta` derive this needs nothing in scope — the generated code
/// names `::delta_struct::Fingerprint` in full — and it accepts enums, which
/// have a perfectly good content hash even though they have no obvious delta.
///
/// ```ignore
/// use delta_struct::Fingerprint;
///
/// #[derive(Fingerprint)]
/// struct Device {
///     services: std::collections::HashSet<String>,
///     online: bool,
/// }
/// ```
#[proc_macro_derive(Fingerprint)]
#[proc_macro_error]
pub fn derive_fingerprint(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        mut generics,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let body = match data {
        Data::Struct(strukt) => {
            // A struct's fields are reached through `self`, by name or by
            // position.
            fingerprint_calls(strukt.fields.iter().enumerate().map(|(i, field)| {
                match &field.ident {
                    Some(ident) => quote!(self.#ident),
                    None => {
                        let index = syn::Index::from(i);
                        quote!(self.#index)
                    }
                }
            }))
        }
        Data::Enum(enom) => {
            // A variant's fields are reached through the locals its pattern
            // binds. The variant's index is folded in first, so two variants
            // holding equal payloads still fingerprint differently.
            let arms = enom.variants.into_iter().enumerate().map(|(index, variant)| {
                let variant_ident = variant.ident;
                let bindings = binding_idents(&variant.fields);
                let pattern = match &variant.fields {
                    Fields::Named(_) => quote!(Self::#variant_ident { #(#bindings),* }),
                    Fields::Unnamed(_) => quote!(Self::#variant_ident( #(#bindings),* )),
                    Fields::Unit => quote!(Self::#variant_ident),
                };
                let fields = fingerprint_calls(bindings.iter().map(|b| quote!(#b)));
                let index = index as u32;
                quote! {
                    #pattern => {
                        ::delta_struct::Fingerprint::fingerprint(&#index, hasher);
                        #fields
                    }
                }
            });
            quote! {
                match self {
                    #(#arms)*
                }
            }
        }
        _ => abort_call_site!(
            "delta_struct::Fingerprint may only be derived for struct and enum types. {} is neither.",
            ident
        ),
    };

    let fingerprint_types = generics
        .type_params()
        .map(|t| t.ident.clone())
        .collect::<Vec<_>>();
    let where_clause = generics.make_where_clause();
    for ty in fingerprint_types {
        where_clause
            .predicates
            .push(syn::parse_quote!(#ty: ::delta_struct::Fingerprint));
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    TokenStream::from(quote! {
        impl #impl_generics ::delta_struct::Fingerprint for #ident #ty_generics #where_clause {
            fn fingerprint(&self, hasher: &mut ::delta_struct::fingerprint::Hasher) {
                #body
            }
        }
    })
}

/// The locals an enum variant's fields bind to in a match pattern: the field's
/// own name where it has one, and `field_0`, `field_1`, … where it does not.
fn binding_idents(fields: &Fields) -> Vec<Ident> {
    fields
        .iter()
        .enumerate()
        .map(|(i, field)| match &field.ident {
            Some(ident) => ident.clone(),
            None => format_ident!("field_{}", i),
        })
        .collect()
}

/// Emits one `Fingerprint::fingerprint` call per expression, in order.
///
/// The expressions name the fields however the caller can reach them —
/// `self.foo` inside a struct, a pattern binding inside a match arm.
fn fingerprint_calls(
    exprs: impl Iterator<Item = proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    let calls = exprs.map(|expr| quote!(::delta_struct::Fingerprint::fingerprint(&#expr, hasher);));
    quote!(#(#calls)*)
}
