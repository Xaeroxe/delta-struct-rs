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
    /// Compared with `!=` and replaced wholesale.
    Scalar,
    /// Diffed recursively via the field type's own `Delta` implementation.
    Delta,
}

const VALID_FIELD_TYPES: &str = "\"ordered\", \"unordered\", \"delta\", or \"scalar\"";

/// Derives `Delta`, generating a `{Self}Delta` struct that holds only the
/// changed parts of a value plus the trait implementation that produces and
/// applies one.
///
/// The generated struct takes the visibility and generic parameters of the
/// type it is derived on, and all of its fields are `pub`. See the
/// [`delta-struct`](https://docs.rs/delta-struct) crate documentation for the
/// full picture, including trait bounds, serde usage, and limitations; what
/// follows is the attribute reference.
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
/// | `delta_leader = "<tokens>"` | Tokens emitted directly above the generated field. On an `unordered` field they land above both the `_add` and the `_remove` field. |
///
/// # Field types
///
/// | Value | Delta representation | Requires |
/// | --- | --- | --- |
/// | `"scalar"` | `Option<T>` | `T: PartialEq` |
/// | `"unordered"` | `{field}_add` and `{field}_remove`, both `Vec<Item>` | `T: IntoIterator + FromIterator<Item> + Extend<Item>`, `Item: PartialEq` |
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
///     services: Vec<String>,
///     online: bool,
/// }
/// ```
///
/// The example is not run as a doctest because this crate cannot depend on the
/// crate that re-exports it; the tested versions live in the `delta-struct`
/// crate documentation.
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
    let delta_struct = quote! {
      #delta_leader
      #vis struct #delta_ident #generics #og_where_clause {
          #delta_fields
      }
    };
    let (delta_compute_let, delta_compute_fields) =
        delta_compute_fields(named, fields.iter().cloned());
    let (delta_apply_let, delta_apply_actions) = delta_apply_fields(named, fields.into_iter());
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
               Some(Self::Output {
                #delta_compute_fields
               })
           } else {
               None
           }
          }

          fn apply_delta(&mut self, delta: Self::Output) {
            let Self::Output {
                #delta_apply_let
            } = delta;
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
/// which of the two it is: tuple struct fields become `field_0`, `field_1`,
/// and so on, because a tuple struct has nowhere to put the `_add`/`_remove`
/// pair an unordered field expands into.
fn delta_fields(
    named: bool,
    iter: impl Iterator<Item = (String, Type, FieldType, String)>,
) -> proc_macro2::TokenStream {
    FromIterator::from_iter(iter.map(|(ident, ty, field_ty, field_leader)| {
        let field_leader = proc_macro2::TokenStream::from_str(&field_leader).unwrap();
        let ident = if named {
            format_ident!("{}", ident)
        } else {
            format_ident!("field_{}", ident)
        };
        match field_ty {
            FieldType::Ordered => {
                quote! {
                    #field_leader
                    pub #ident: ::delta_struct::SeqDelta<<#ty as ::std::iter::IntoIterator>::Item>,
                }
            }
            FieldType::Unordered => {
                let add = format_ident!("{}_add", ident);
                let remove = format_ident!("{}_remove", ident);
                quote! {
                 #field_leader
                 pub #add: Vec<<#ty as ::std::iter::IntoIterator>::Item>,
                 #field_leader
                 pub #remove: Vec<<#ty as ::std::iter::IntoIterator>::Item>,
                }
            }
            FieldType::Scalar => {
                quote! {
                  #field_leader
                  pub #ident: ::std::option::Option<#ty>,
                }
            }
            FieldType::Delta => {
                quote! {
                    #field_leader
                    pub #ident: ::std::option::Option<<#ty as Delta>::Output>,
                }
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
    iter: impl Iterator<Item = (String, Type, FieldType, String)>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    iter.map(|(og_ident, _ty, field_ty, _field_leader)| {
        let ident = if named {
            format_ident!("{}", og_ident)
        } else {
            format_ident!("field_{}", og_ident)
        };
        let og_ident: proc_macro2::TokenStream = FromStr::from_str(&og_ident).unwrap();
        match field_ty {
            FieldType::Ordered => (
                quote! {
                    let #ident = ::delta_struct::seq::diff(old.#og_ident, new.#og_ident);
                    delta_is_some = delta_is_some || !#ident.is_empty();
                },
                quote! {
                    #ident,
                },
            ),
            FieldType::Unordered => {
                let add = format_ident!("{}_add", ident);
                let remove = format_ident!("{}_remove", ident);

                (
                    quote! {
                        // Start from every item in `new` and cancel out the
                        // ones `old` also had, one occurrence at a time, so
                        // duplicates are counted rather than deduplicated.
                        // Whatever is left in `old` is what was removed.
                        let mut #add = new.#og_ident.into_iter().collect::<::std::vec::Vec<_>>();
                        let #remove = old.#og_ident.into_iter().filter_map(|i| {
                            if let Some(index) = #add.iter().position(|a| a == &i) {
                                #add.remove(index);
                                None
                            } else {
                                Some(i)
                            }
                        }).collect::<::std::vec::Vec<_>>();
                        delta_is_some = delta_is_some || !#add.is_empty() || !#remove.is_empty();
                    },
                    quote! {
                        #add,
                        #remove,
                    },
                )
            }
            FieldType::Scalar => (
                quote! {
                   let #ident = if old.#og_ident != new.#og_ident {
                       delta_is_some = true;
                       Some(new.#og_ident)
                   } else {
                       None
                   };
                },
                quote! {
                    #ident,
                },
            ),
            FieldType::Delta => (
                quote! {
                    let #ident = Delta::delta(old.#og_ident, new.#og_ident);
                    delta_is_some = delta_is_some || #ident.is_some();

                },
                quote! {
                    #ident,
                },
            ),
        }
    })
    .unzip()
}

/// Emits the body of `Delta::apply_delta`, as `(destructuring pattern,
/// statements)`.
///
/// The pattern takes the delta struct apart into locals — binding the
/// `_remove` lists as `mut`, since applying them consumes their entries — and
/// the statements write each change back into `self`. Fields arrive in the
/// same shape as in [`delta_fields`].
fn delta_apply_fields(
    named: bool,
    iter: impl Iterator<Item = (String, Type, FieldType, String)>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    iter.map(|(og_ident, ty, field_ty, _field_leader)| {
        let ident = if named {
            format_ident!("{}", og_ident)
        } else {
            format_ident!("field_{}", og_ident)
        };
        let og_ident: proc_macro2::TokenStream = FromStr::from_str(&og_ident).unwrap();
        match field_ty {
            FieldType::Ordered => (
                quote! {
                    #ident,
                },
                quote! {
                    ::delta_struct::seq::apply(&mut self.#og_ident, #ident);
                },
            ),
            FieldType::Unordered => {
                let add = format_ident!("{}_add", ident);
                let remove = format_ident!("{}_remove", ident);
                (
                    quote! {
                        #add,
                        mut #remove,
                    },
                    quote! {
                        {
                            // Take the collection by value so its items can be
                            // moved through the filter, then rebuild it minus
                            // the removals and plus the additions. Note that
                            // this does not preserve position for ordered
                            // collections — additions land at the end.
                            let og = ::std::mem::replace(&mut self.#og_ident, ::std::iter::FromIterator::from_iter(vec![]));
                            let mut #ident: #ty = ::std::iter::FromIterator::from_iter(og.into_iter().filter_map(|i| {
                               if let Some(index) = #remove.iter().position(|a| a == &i) {
                                 #remove.remove(index);
                                 None
                               } else {
                                 Some(i)
                               }
                            }));
                            #ident.extend(#add.into_iter());
                            self.#og_ident = #ident;
                        }
                    }
                )
            }
            FieldType::Scalar =>
            (
                quote! {
                    #ident,
                },
                quote! {
                   if let Some(v) = #ident {
                       self.#og_ident = v;
                   }
                }
            ),
            FieldType::Delta =>
            (
                quote! {
                    #ident,
                },
                quote!{
                   if let Some(v) = #ident {
                       self.#og_ident.apply_delta(v);
                   }
                }
            ),
        }
    }).unzip()
}

/// Resolves each field's parsed attributes against the container default,
/// collecting *every* bad field rather than stopping at the first, so one
/// compile reports them all.
#[allow(clippy::manual_try_fold)] // Collects errors too
fn collect_results(
    iter: impl Iterator<
        Item = (
            String,
            Type,
            Result<(Option<FieldType>, String), FieldTypeError>,
        ),
    >,
    default_field_type: FieldType,
) -> Result<Vec<(String, Type, FieldType, String)>, Vec<String>> {
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
fn get_fieldtype_from_attrs(
    iter: impl Iterator<Item = Attribute>,
    attr_name: &str,
) -> Result<(Option<FieldType>, String), FieldTypeError> {
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
        "scalar" => Some(FieldType::Scalar),
        "delta" => Some(FieldType::Delta),
        _ => None,
    }
}
