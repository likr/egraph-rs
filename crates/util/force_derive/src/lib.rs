extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemStruct};

#[proc_macro_derive(Force)]
pub fn derive_self_name(input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as ItemStruct);
    let struct_name = item.ident;
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
    let gen = quote! {
        impl #impl_generics Force for #struct_name #ty_generics #where_clause {
            fn apply(&self, points: &mut [Point], alpha: f32) {
                for u in 0..points.len() {
                    self.apply_to_node(u, points, alpha);
                }
            }
        }
    };
    gen.into()
}
