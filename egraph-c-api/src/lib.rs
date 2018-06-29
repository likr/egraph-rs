extern crate clp;

#[no_mangle]
pub fn hoge() {
    let mut model = clp::Model::new();
    model.resize(3, 3);
    println!("{} {}", model.number_rows(), model.number_columns());
}
