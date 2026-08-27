//! Procedural macro implementation behind the `delta-struct` crate.
//!
//! Use [`delta-struct`](https://docs.rs/delta-struct) rather than depending on
//! this crate directly; it re-exports the [`Delta`] derive alongside the trait
//! the derive generates an implementation of, and carries the user-facing
//! documentation.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2, TokenTree};
use proc_macro_error::{abort_call_site, proc_macro_error};
use quote::{format_ident, quote, ToTokens};
use std::{
    fmt::{self, Display},
    iter::FromIterator,
    str::FromStr,
};
use syn::{
    parse_macro_input, punctuated::Punctuated, Attribute, Data, DeriveInput, Expr, ExprLit, Fields,
    Ident, Lit, Meta, MetaList, MetaNameValue, Path, PredicateType, Token, TraitBound,
    TraitBoundModifiers, Type, TypeParamBound, WherePredicate,
};

/// How a single field is diffed, and therefore how it is represented on the
/// generated delta struct.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum FieldType {
    /// A positional diff: the delta is a Myers edit script over the sequence.
    Ordered,
    /// A bag of items: the delta records additions and removals, not order.
    /// Its shape is the collection's to choose, through `Unordered`, since a
    /// map can name a departing entry by key where a set cannot.
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
/// The generated type takes the visibility and generic parameters of the type
/// it is derived on, and mirrors its shape: a tuple struct's delta is a tuple
/// struct with its fields in the same positions, and an enum's is an enum with
/// one variant per *diffable* source variant. A struct's delta fields are all
/// `pub`.
///
/// For an enum, `Output` is `EnumDelta<Self, {Self}Delta>` rather than the
/// bare companion, because a value can change variant as well as change within
/// one — and changing variant is a replacement rather than a difference.
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
/// | `"scalar"` | `delta_struct::ScalarDelta<T>` | `T: PartialEq` |
/// | `"unordered"` | `<T as Unordered>::Delta` — `BagDelta<Item>` for a set, `EntryDelta<Key, Value>` for a map | `T: Unordered` |
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
            Err(e) => {
                abort_call_site!(
                    "delta_struct(default = ...) for {} is not an accepted value, expected {}. {}",
                    ident,
                    VALID_FIELD_TYPES,
                    e,
                );
            }
        };

    let delta_leader = match proc_macro2::TokenStream::from_str(&delta_leader) {
        Ok(v) => v,
        Err(e) => {
            abort_call_site!("error parsing delta leader as token stream {}", e);
        }
    };
    let delta_ident = format_ident!("{}Delta", ident);
    // The delta type repeats the source type's generics verbatim, bounds and
    // all, since its fields can project through them — `<T as Delta>::Output`
    // for a delta field, `<T as IntoIterator>::Item` for an unordered one. Grab
    // the where clause before the `PartialEq` predicates below are pushed onto
    // it; those are the impl's business, not the type's.
    let og_where_clause = generics.where_clause.clone();
    let ty_generics_only = generics.split_for_impl().1.to_token_stream();

    let Generated {
        delta_type,
        output_ty,
        delta_body,
        apply_body,
    } = match data {
        Data::Struct(strukt) => struct_impl(
            &ident,
            &vis,
            &delta_ident,
            &delta_leader,
            &generics,
            &og_where_clause,
            &ty_generics_only,
            strukt.fields,
            default_field_type,
        ),
        Data::Enum(enom) => enum_impl(
            &ident,
            &vis,
            &delta_ident,
            &delta_leader,
            &generics,
            &og_where_clause,
            &ty_generics_only,
            enom.variants.into_iter().collect(),
            default_field_type,
        ),
        Data::Union(_) => {
            abort_call_site!(
                "delta_struct::Delta may only be derived for struct and enum types. {} is a union.",
                ident
            )
        }
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
            modifiers: TraitBoundModifiers::default(),
            lifetimes: None,
            maybe: None,
            path: Path {
                leading_colon: Some(Token!(::)(Span::call_site())),
                segments,
            },
        }));
        where_clause
            .predicates
            .push(WherePredicate::Type(PredicateType {
                attrs: Vec::new(),
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
          type Output = #output_ty;

          fn delta(old: Self, new: Self) -> Option<Self::Output> {
              #delta_body
          }

          // A one-variant enum has no mismatch to catch, and an enum of only
          // unit variants has an uninhabited delta, which makes the tail of
          // this unreachable. Both are fine; neither should warn the caller.
          #[allow(unreachable_patterns, unreachable_code)]
          fn apply_delta(
              &mut self,
              delta: Self::Output,
          ) -> ::std::result::Result<(), ::delta_struct::Mismatch> {
              #apply_body
          }
      }
    };
    let output = quote! {
        #delta_type

        #delta_impl
    };
    TokenStream::from(output)
}

/// The four pieces the struct and enum paths each produce: the delta type's
/// declaration, the `Output` it becomes, and the two method bodies.
struct Generated {
    delta_type: TokenStream2,
    output_ty: TokenStream2,
    delta_body: TokenStream2,
    apply_body: TokenStream2,
}

/// Reads one group of source fields into the shape the code generators want,
/// resolving each against the container's default field type.
///
/// Returns `(named, fields)`, where `named` says whether the group is written
/// with braces or with parentheses.
fn read_fields(owner: &Ident, fields: Fields, default_field_type: FieldType) -> (bool, Vec<Field>) {
    let (named, collected) = match fields {
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
    };
    match collected {
        Ok(fields) => (named, fields),
        Err(bad_fields) => {
            let bad_fields = format!("{:?}", bad_fields);
            abort_call_site!(
                "delta_struct(field_type = ...) for fields in {}: {} are not valid values. Expected {}.",
                owner,
                bad_fields,
                VALID_FIELD_TYPES
            )
        }
    }
}

/// Generates the delta of a struct: a companion struct of the same shape, and
/// two method bodies that walk its fields.
#[allow(clippy::too_many_arguments)] // All of it is one type's description.
fn struct_impl(
    ident: &Ident,
    vis: &syn::Visibility,
    delta_ident: &Ident,
    delta_leader: &TokenStream2,
    generics: &syn::Generics,
    og_where_clause: &Option<syn::WhereClause>,
    ty_generics: &TokenStream2,
    fields: Fields,
    default_field_type: FieldType,
) -> Generated {
    let (named, fields) = read_fields(ident, fields, default_field_type);
    let delta_fields = delta_fields(named, fields.iter().cloned());
    let (compute_let, compute_fields) =
        delta_compute_fields(named, Source::Whole, fields.iter().cloned());
    let (apply_let, apply_actions) = delta_apply_fields(named, Source::Whole, fields.into_iter());

    // A tuple struct's delta is a tuple struct too, which means the
    // declaration, the initializer, and the destructuring pattern all have to
    // switch from braces to parentheses together. Two things differ beyond the
    // brackets: a tuple struct puts its `where` clause *after* the fields and
    // ends in a semicolon, and its constructor lives in the value namespace,
    // which `Self::Output` — an associated type — cannot reach, so the
    // initializer and pattern name the struct itself and let inference supply
    // its generics.
    let (delta_type, compute_init, apply_pattern) = if named {
        (
            quote! {
                #delta_leader
                #vis struct #delta_ident #generics #og_where_clause {
                    #delta_fields
                }
            },
            quote!(Self::Output { #compute_fields }),
            quote!(Self::Output { #apply_let }),
        )
    } else {
        (
            quote! {
                #delta_leader
                #vis struct #delta_ident #generics (#delta_fields) #og_where_clause;
            },
            quote!(#delta_ident(#compute_fields)),
            quote!(#delta_ident(#apply_let)),
        )
    };

    Generated {
        delta_type,
        output_ty: quote!(#delta_ident #ty_generics),
        delta_body: quote! {
            let mut delta_is_some = false;
            #compute_let
            if delta_is_some {
                Some(#compute_init)
            } else {
                None
            }
        },
        // A struct's delta always fits, so this is the arm of `apply_delta`
        // that can only ever be `Ok` — the `?`s inside come from fields whose
        // own types are enums.
        apply_body: quote! {
            let #apply_pattern = delta;
            #apply_actions
            Ok(())
        },
    }
}

/// Generates the delta of an enum.
///
/// The companion enum carries one variant per *diffable* source variant — a
/// field-less variant can never differ from itself, so giving it an arm would
/// only create one nothing could construct. Changing variant is not a
/// difference at all but a replacement, and that case lives in
/// [`EnumDelta::Became`](::delta_struct::EnumDelta), a type in the runtime
/// crate rather than an arm here, so it cannot collide with a variant the user
/// wrote.
#[allow(clippy::too_many_arguments)] // All of it is one type's description.
fn enum_impl(
    ident: &Ident,
    vis: &syn::Visibility,
    delta_ident: &Ident,
    delta_leader: &TokenStream2,
    generics: &syn::Generics,
    og_where_clause: &Option<syn::WhereClause>,
    ty_generics: &TokenStream2,
    variants: Vec<syn::Variant>,
    default_field_type: FieldType,
) -> Generated {
    if variants.is_empty() {
        abort_call_site!(
            "delta_struct::Delta cannot be derived for {}, which has no variants: an \
             uninhabited type has no two values to differ.",
            ident
        )
    }

    let read: Vec<(Ident, bool, Vec<Field>)> = variants
        .into_iter()
        .map(|variant| {
            let (named, fields) = read_fields(ident, variant.fields, default_field_type);
            (variant.ident, named, fields)
        })
        .collect();

    let mut delta_variants = Vec::new();
    let mut diff_arms = Vec::new();
    let mut apply_arms = Vec::new();

    for (variant, named, fields) in &read {
        if fields.is_empty() {
            // Nothing to diff, and nothing to apply: two of these are equal by
            // being the same variant.
            let pattern = variant_pattern(&quote!(Self), variant, *named, fields, Some("old"));
            diff_arms.push(quote!((#pattern, Self::#variant) => None,));
            continue;
        }

        // Enum variant fields carry the enum's visibility, so unlike a struct's
        // they must not be written `pub`.
        let declared = delta_fields_inner(*named, false, fields.iter().cloned());
        delta_variants.push(if *named {
            quote!(#variant { #declared })
        } else {
            quote!(#variant(#declared))
        });

        let (compute_let, compute_fields) =
            delta_compute_fields(*named, Source::Bound, fields.iter().cloned());
        let old = variant_pattern(&quote!(Self), variant, *named, fields, Some("old"));
        let new = variant_pattern(&quote!(Self), variant, *named, fields, Some("new"));
        let init = if *named {
            quote!(#delta_ident::#variant { #compute_fields })
        } else {
            quote!(#delta_ident::#variant(#compute_fields))
        };
        diff_arms.push(quote! {
            (#old, #new) => {
                let mut delta_is_some = false;
                #compute_let
                if delta_is_some {
                    Some(::delta_struct::EnumDelta::Delta(#init))
                } else {
                    None
                }
            }
        });

        let (_, apply_actions) = delta_apply_fields(*named, Source::Bound, fields.iter().cloned());
        let target = variant_pattern(&quote!(Self), variant, *named, fields, Some("self"));
        let carried = variant_pattern(&quote!(#delta_ident), variant, *named, fields, None);
        apply_arms.push(quote! {
            (#target, #carried) => { #apply_actions }
        });
    }

    // Both halves of a mismatch report a name, and both are found by matching
    // — `{ .. }` fits every variant shape, so one arm per variant does it.
    let source_names = read
        .iter()
        .map(|(variant, ..)| quote!(Self::#variant { .. } => stringify!(#variant),));
    let delta_names = read
        .iter()
        .filter(|(_, _, fields)| !fields.is_empty())
        .map(|(variant, ..)| quote!(#delta_ident::#variant { .. } => stringify!(#variant),));

    Generated {
        delta_type: quote! {
            #delta_leader
            #vis enum #delta_ident #generics #og_where_clause {
                #(#delta_variants,)*
            }
        },
        output_ty: quote! {
            ::delta_struct::EnumDelta<#ident #ty_generics, #delta_ident #ty_generics>
        },
        delta_body: quote! {
            #[allow(unreachable_patterns)] // A one-variant enum never `Became`.
            match (old, new) {
                #(#diff_arms)*
                // Different variants: there is no difference to describe, only
                // a replacement.
                (_, new) => Some(::delta_struct::EnumDelta::Became(new)),
            }
        },
        apply_body: quote! {
            let delta = match delta {
                ::delta_struct::EnumDelta::Became(new) => {
                    *self = new;
                    return Ok(());
                }
                ::delta_struct::EnumDelta::Delta(delta) => delta,
            };
            match (&mut *self, delta) {
                #(#apply_arms)*
                (found, mismatched) => {
                    return Err(::delta_struct::Mismatch {
                        type_name: stringify!(#ident),
                        expected: match mismatched { #(#delta_names)* },
                        found: match found { #(#source_names)* },
                    });
                }
            }
            Ok(())
        },
    }
}

/// The pattern that takes one variant apart, binding each field to a local.
///
/// `prefix` distinguishes the several copies of a variant that appear in one
/// match — `old_`, `new_`, `self_` — or is [`None`] for the delta being
/// consumed, whose fields bind to the bare local names the generated field
/// code already refers to.
fn variant_pattern(
    path: &TokenStream2,
    variant: &Ident,
    named: bool,
    fields: &[Field],
    prefix: Option<&str>,
) -> TokenStream2 {
    if fields.is_empty() {
        return quote!(#path::#variant);
    }
    let bindings = fields
        .iter()
        .map(|(og_ident, ..)| {
            let local = local_ident(named, og_ident);
            match prefix {
                Some(prefix) => format_ident!("{}_{}", prefix, local),
                None => local,
            }
        })
        .collect::<Vec<_>>();
    if named {
        let names = fields
            .iter()
            .map(|(og_ident, ..)| format_ident!("{}", og_ident));
        quote!(#path::#variant { #(#names: #bindings),* })
    } else {
        quote!(#path::#variant( #(#bindings),* ))
    }
}

/// Emits the field declarations of the generated delta struct.
///
/// Fields arrive as `(name, type, field type, delta_leader)`, where `name` is
/// the source field's name or, for tuple structs, its index. `named` says
/// which of the two it is, and so whether these declarations are about to be
/// wrapped in braces or in parentheses: a tuple struct's delta is a tuple
/// struct too, and its fields are positional rather than named.
fn delta_fields(named: bool, iter: impl Iterator<Item = Field>) -> proc_macro2::TokenStream {
    delta_fields_inner(named, true, iter)
}

/// The body of [`delta_fields`], with a say over `pub`.
///
/// A struct's delta fields are all `pub`; an enum variant's take the enum's
/// visibility and may not be written `pub` at all.
fn delta_fields_inner(
    named: bool,
    public: bool,
    iter: impl Iterator<Item = Field>,
) -> proc_macro2::TokenStream {
    let vis = public.then(|| quote!(pub));
    FromIterator::from_iter(iter.map(|(ident, ty, field_ty, field_leader)| {
        let field_leader = proc_macro2::TokenStream::from_str(&field_leader).unwrap();
        let declared_ty = match field_ty {
            FieldType::Ordered => {
                quote!(::delta_struct::SeqDelta<<#ty as ::std::iter::IntoIterator>::Item>)
            }
            FieldType::Unordered => {
                // Unlike the other collection field types this does not name a
                // delta type directly: a set's membership diff and a map's are
                // different shapes, and `Unordered` is what picks between them.
                quote!(<#ty as ::delta_struct::Unordered>::Delta)
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
            FieldType::Scalar => quote!(::delta_struct::ScalarDelta<#ty>),
            FieldType::Delta => quote!(::std::option::Option<<#ty as Delta>::Output>),
        };
        if named {
            let ident = format_ident!("{}", ident);
            quote! {
                #field_leader
                #vis #ident: #declared_ty,
            }
        } else {
            quote! {
                #field_leader
                #vis #declared_ty,
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
    source: Source,
    iter: impl Iterator<Item = Field>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    iter.map(|(og_ident, ty, field_ty, _field_leader)| {
        let ident = local_ident(named, &og_ident);
        let (old, new) = source.sides(&og_ident, &ident);
        let statements = match field_ty {
            FieldType::Ordered | FieldType::UnorderedDelta => {
                let module = collection_module(field_ty);
                quote! {
                    let #ident = ::delta_struct::#module::diff(#old, #new);
                    delta_is_some = delta_is_some || !#ident.is_empty();
                }
            }
            // `Unordered::diff` reports "nothing changed" as `None` rather than
            // as an empty delta, so this reads like the `Scalar` arm below
            // rather than like the two collection modules above. The field
            // still holds an empty delta, which is what `Default` supplies.
            FieldType::Unordered => quote! {
                let #ident = match <#ty as ::delta_struct::Unordered>::diff(#old, #new) {
                    Some(v) => {
                        delta_is_some = true;
                        v
                    }
                    None => ::std::default::Default::default(),
                };
            },
            FieldType::Scalar => quote! {
                let #ident = if #old != #new {
                    delta_is_some = true;
                    ::delta_struct::ScalarDelta::Changed(#new)
                } else {
                    ::delta_struct::ScalarDelta::Unchanged
                };
            },
            FieldType::Delta => quote! {
                let #ident = Delta::delta(#old, #new);
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
    source: Source,
    iter: impl Iterator<Item = Field>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    iter.map(|(og_ident, ty, field_ty, _field_leader)| {
        let ident = local_ident(named, &og_ident);
        let target = source.target(&og_ident, &ident);
        let statements = match field_ty {
            // `map::apply` is the one collection helper that can fail, because
            // it is the one that recurses into `apply_delta`.
            FieldType::Ordered | FieldType::UnorderedDelta => {
                let module = collection_module(field_ty);
                let question = (field_ty == FieldType::UnorderedDelta).then(|| quote!(?));
                quote! {
                    ::delta_struct::#module::apply(&mut #target, #ident)#question;
                }
            }
            FieldType::Unordered => quote! {
                <#ty as ::delta_struct::Unordered>::apply(&mut #target, #ident);
            },
            FieldType::Scalar => quote! {
                if let ::delta_struct::ScalarDelta::Changed(v) = #ident {
                    #target = v;
                }
            },
            FieldType::Delta => quote! {
                if let Some(v) = #ident {
                    #target.apply_delta(v)?;
                }
            },
        };
        // Binds one local per field, in declaration order — see the matching
        // note in `delta_compute_fields` about braces versus parens.
        (quote!(#ident,), statements)
    })
    .unzip()
}

/// How generated code reaches the two sides of a field.
///
/// A struct's impl holds whole values and reads through them. An enum's has
/// taken its values apart in a match pattern, so the fields are already locals
/// by the time the per-field code runs — and for `apply_delta` they are `&mut`
/// locals, which is why [`Source::target`] dereferences them.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Source {
    /// Through the values themselves: `old.foo`, `new.foo`, `self.foo`.
    Whole,
    /// Through pattern bindings: `old_foo`, `new_foo`, `*self_foo`.
    Bound,
}

impl Source {
    /// The expressions naming a field's old and new values in `Delta::delta`.
    fn sides(self, og_ident: &str, ident: &Ident) -> (TokenStream2, TokenStream2) {
        match self {
            Source::Whole => {
                let og_ident = field_accessor(og_ident);
                (quote!(old.#og_ident), quote!(new.#og_ident))
            }
            Source::Bound => {
                let old = format_ident!("old_{}", ident);
                let new = format_ident!("new_{}", ident);
                (quote!(#old), quote!(#new))
            }
        }
    }

    /// The place expression a field is written back to in `apply_delta`.
    fn target(self, og_ident: &str, ident: &Ident) -> TokenStream2 {
        match self {
            Source::Whole => {
                let og_ident = field_accessor(og_ident);
                quote!(self.#og_ident)
            }
            Source::Bound => {
                let binding = format_ident!("self_{}", ident);
                quote!((*#binding))
            }
        }
    }
}

/// A source field's name as it is written after a `.` — its identifier, or the
/// bare index of a tuple field.
fn field_accessor(og_ident: &str) -> TokenStream2 {
    FromStr::from_str(og_ident).unwrap()
}

/// The local a generated field binds to: its own name, or `field_0`,
/// `field_1`, … where the source field is positional.
fn local_ident(named: bool, og_ident: &str) -> Ident {
    if named {
        format_ident!("{}", og_ident)
    } else {
        format_ident!("field_{}", og_ident)
    }
}

/// The runtime module backing a collection field type whose delta type is
/// fixed by the field type alone.
///
/// These two differ in what their delta looks like but not in how the derive
/// drives one: each module pairs a `diff` and an `apply` over a delta type
/// that reports whether it is empty. `unordered` is not among them — its shape
/// depends on the collection rather than the field type, so it goes through
/// the `Unordered` trait instead. Panics for every field type the callers
/// never pass.
fn collection_module(field_ty: FieldType) -> Ident {
    match field_ty {
        FieldType::Ordered => format_ident!("seq"),
        FieldType::UnorderedDelta => format_ident!("map"),
        FieldType::Unordered | FieldType::Scalar | FieldType::Delta => {
            unreachable!("{:?} does not have a fixed collection module", field_ty)
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
    Syn(syn::Error),
    /// The `delta_struct(...)` attribute contained entries that were not
    /// `name = "value"` pairs.
    UnrecognizedJunkFound(Vec<Meta>),
}

impl From<syn::Error> for FieldTypeError {
    fn from(value: syn::Error) -> Self {
        FieldTypeError::Syn(value)
    }
}

impl Display for FieldTypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FieldTypeError::Syn(error) => write!(f, "{error}"),
            FieldTypeError::UnrecognizedJunkFound(metas) => {
                let metas = metas
                    .iter()
                    .map(|m| m.into_token_stream().to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                write!(
                    f,
                    "expected a comma separated list of named values, got {metas}"
                )
            }
        }
    }
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
        if let Meta::List(MetaList { path, .. }) = &attr.meta {
            let Path { segments, .. } = path;
            if segments
                .iter()
                .map(|p| &p.ident)
                .eq(["delta_struct"].iter().cloned())
            {
                let nested =
                    attr.parse_args_with(Punctuated::<Meta, Token!(,)>::parse_terminated)?;
                let values: Result<Vec<_>, Vec<Meta>> = nested
                    .iter()
                    .map(|meta| match meta {
                        Meta::NameValue(MetaNameValue {
                            path,
                            value:
                                Expr::Lit(ExprLit {
                                    lit: Lit::Str(s), ..
                                }),
                            ..
                        }) => Ok((path.get_ident().map(|i| i.to_string()), s.value())),
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
                let v = values.map_err(FieldTypeError::UnrecognizedJunkFound)?;
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
                return Ok((field_type, delta_leader));
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
