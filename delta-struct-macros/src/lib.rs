extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::{TokenTree, Span};
use quote::{format_ident, quote};
use std::{str::FromStr, iter::FromIterator};
use syn::{
   Ident, parse_macro_input, Attribute, Data, DeriveInput, Fields, Lit, LitStr, Meta, MetaNameValue, MetaList, NestedMeta,
    Path, Type, WherePredicate, Token, PredicateType, punctuated::Punctuated, TypeParamBound, TraitBoundModifier, TraitBound 
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum FieldType {
    Ordered,
    Unordered,
    Scalar,
}

const VALID_FIELD_TYPES: &str = "\"ordered\", \"unordered\", or \"scalar\"";

#[proc_macro_derive(Delta, attributes(delta_struct))]
pub fn derive_delta(input: TokenStream) -> TokenStream {
    let DeriveInput {
        attrs,
        vis,
        ident,
        mut generics,
        data,
    } = parse_macro_input!(input as DeriveInput);
    let default_field_type = match get_fieldtype_from_attrs(attrs.into_iter(), "default") {
        Ok(v) => v.unwrap_or(FieldType::Scalar),
        Err(_) => {
            let ident = LitStr::new(&ident.to_string(), ident.span());
            return quote!(
                ::std::compile_error!(
                    ::std::concat!("delta_struct(default = ...) for ", stringify!(#ident), " is not an accepted value, expected ", stringify!(#VALID_FIELD_TYPES)".")
                    )
                )
                .into();
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
            Fields::Unit => {
                let ident = format_ident!("{}", ident);
                return quote!(::std::compile_error!(::std::concat!(
                    "delta_struct::Delta can't be implemented for unit struct, ",
                    stringify!(#ident),
                    ", there is nothing to diff."
                )))
                .into();
            }
        },
        _ => {
            let ident = format_ident!("{}", ident);
            return quote!(::std::compile_error!(::std::concat!(
                "delta_struct::Delta may only be derived for struct types currently. ",
                stringify!(#ident),
                " is not a struct type. Sorry."
            )))
            .into();
        }
    };
    let fields = match fields {
        Ok(fields) => fields,
        Err(bad_fields) => {
            let ident = LitStr::new(&ident.to_string(), ident.span());
            let bad_fields = LitStr::new(&format!("{:?}", bad_fields), Span::call_site());
            return quote!(::std::compile_error!(::std::concat!(
                "delta_struct(field_type = ...) for fields in ",
                stringify!(#ident),
                ": ",
                stringify!(#bad_fields),
                " are not valid values. Expected ",
                stringify!(#VALID_FIELD_TYPES),
                "."
            )))
            .into();
        }
    };
    let delta_ident = format_ident!("{}Delta", ident);
    let delta_fields = delta_fields(named, fields.iter().cloned());
    let delta_struct = quote! {
      #vis struct #delta_ident #generics {
          #delta_fields
      }
    };
    let (delta_compute_let, delta_compute_fields) = delta_compute_fields(named, fields.into_iter());
    let partial_eq_types = generics.type_params().map(|t| t.ident.clone()).collect::<Vec<_>>();
    let where_clause = generics.make_where_clause();
    for ty in partial_eq_types {
        let mut bounds = Punctuated::new();
        let mut segments = Punctuated::new();
        segments.push(Ident::new("std", Span::call_site()).into());
        segments.push(Ident::new("cmp", Span::call_site()).into());
        segments.push(Ident::new("PartialEq", Span::call_site()).into());
        bounds.push(
            TypeParamBound::Trait(
                TraitBound {
                    paren_token: None,
                    modifier: TraitBoundModifier::None,
                    lifetimes: None,
                    path: Path {
                        leading_colon: Some(Token!(::)(Span::call_site())),
                        segments
                    }, 
                }
            )
        );
        where_clause
            .predicates
            .push(
                WherePredicate::Type(
                    PredicateType {
                        lifetimes: None,
                        bounded_ty: Type::Verbatim(
                            <Ident as Into<TokenTree>>::into(ty).into()
                        ),
                        colon_token: Token!(:)(Span::call_site()),
                        bounds
                    }
                )
            );
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let delta_impl = quote! {
      impl #impl_generics Delta for #ident #ty_generics #where_clause  {
          type Output = #delta_ident #generics;

          fn delta(old: Self, new: Self) -> Self::Output {
           #delta_compute_let
           Self::Output {
            #delta_compute_fields
           }
          }
      }
    };
    let output = quote! {
        #delta_struct

        #delta_impl
    };
    TokenStream::from(output)
}

fn delta_fields(
    named: bool,
    iter: impl Iterator<Item = (String, Type, FieldType)>,
) -> proc_macro2::TokenStream {
    FromIterator::from_iter(iter.map(|(ident, ty, field_ty)| {
        let ident = if named {
            format_ident!("{}", ident)
        } else {
            format_ident!("field_{}", ident)
        };
        match field_ty {
            FieldType::Ordered => unimplemented!(),
            FieldType::Unordered => {
                let add = format_ident!("{}_add", ident);
                let remove = format_ident!("{}_remove", ident);
                quote! {
                 #add: Vec<<#ty as ::std::iter::IntoIterator>::Item>,
                 #remove: Vec<<#ty as ::std::iter::IntoIterator>::Item>,
                }
            }
            FieldType::Scalar => {
                quote! {
                  #ident: ::std::option::Option<#ty>,
                }
            }
        }
    }))
}

fn delta_compute_fields(
    named: bool,
    iter: impl Iterator<Item = (String, Type, FieldType)>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    iter.map(|(og_ident, _ty, field_ty)| {
        let ident = if named {
                    format_ident!("{}", og_ident)
                } else {
                    format_ident!("field_{}", og_ident)
                };
        let og_ident: proc_macro2::TokenStream = FromStr::from_str(&og_ident).unwrap();
        match field_ty {
            FieldType::Ordered => unimplemented!(),
            FieldType::Unordered => {
                let add = format_ident!("{}_add", ident);
                let remove = format_ident!("{}_remove", ident);

                (
                    quote! {
                        let mut #add = new.#og_ident.into_iter().collect::<::std::vec::Vec<_>>();
                        let mut in_both = ::std::vec![];
                        let mut #remove = old.#og_ident.into_iter().filter_map(|i| {
                            if #add.iter().any(|a| a == &i) {
                                in_both.push(i);
                                None
                            } else {
                                Some(i)
                            }
                        }).collect::<::std::vec::Vec<_>>();
                        #add.retain(|i| !in_both.iter().any(|a| a == i));
                    },
                    quote! {
                        #add,
                        #remove,
                    }
                )

            }
            FieldType::Scalar => {
                (
                    quote!(),
                    quote! {
                        #ident: if old.#og_ident != new.#og_ident { Some(new.#og_ident) } else { None },
                    } 
                )
            }
        }
    })
    .unzip()
}

fn collect_results(
    iter: impl Iterator<Item = (String, Type, Result<Option<FieldType>, FieldTypeError>)>,
    default_field_type: FieldType,
) -> Result<Vec<(String, Type, FieldType)>, Vec<String>> {
    iter.fold(Ok(vec![]), |v, i| match (v, i) {
        (Ok(mut v), (ident, b, Ok(c))) => {
            v.push((ident, b, c.unwrap_or(default_field_type)));
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
    WrongNameOrTooMuch(Vec<(Option<String>, String)>),
    UnrecognizedJunkFound(Vec<NestedMeta>),
}

fn get_fieldtype_from_attrs(
    iter: impl Iterator<Item = Attribute>,
    attr_name: &str,
) -> Result<Option<FieldType>, FieldTypeError> {
    for attr in iter {
        if let Ok(Meta::List(MetaList { path, nested, .. })) = attr.parse_meta() {
            let Path { segments, .. } = path;
            if segments
                .iter()
                .map(|p| &p.ident)
                .eq(["delta_struct"].iter().cloned())
            {
                let values: Result<Vec<_>, Vec<NestedMeta>> = nested.iter().map(|nested_meta| match nested_meta {
                    NestedMeta::Meta(Meta::NameValue(MetaNameValue {path, lit: Lit::Str(s), ..})) => Ok((path.get_ident().map(|i| i.to_string()), s.value())),
                    e @ _ => Err(e)
                }).fold(Ok(vec![]), |v, i| match (v, i) {
                    (Ok(mut v), Ok(i)) => {
                        v.push(i);
                        Ok(v)
                    },
                    (Ok(_), Err(e)) => Err(vec![e.clone()]),
                    (Err(mut v), Err(e)) => {
                        v.push(e.clone());
                        Err(v)
                    },
                    (v @ Err(_), _) => v

                });
                return 
                   match values {
                    Ok(v) => {
                        if v.len() == 1 && v[0].0.as_deref() == Some(attr_name) {
                            Ok(string_to_fieldtype(&v[0].1))
                        } else {
                            Err(FieldTypeError::WrongNameOrTooMuch(v))
                        }
                    },
                    Err(v) => Err(FieldTypeError::UnrecognizedJunkFound(v))
                   }
            }
        }
    }
    Ok(None)
}

fn string_to_fieldtype(s: &str) -> Option<FieldType> {
    match s {
        "ordered" => Some(FieldType::Ordered),
        "unordered" => Some(FieldType::Unordered),
        "scalar" => Some(FieldType::Scalar),
        _ => None,
    }
}
