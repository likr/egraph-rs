extern crate clp_sys;

#[test]
fn test_clp_version_major() {
    unsafe {
        let c_buf = clp_sys::Clp_Version();
        let c_str = std::ffi::CStr::from_ptr(c_buf);
        let version = c_str.to_str().unwrap();
        println!("{}", version);
    }
}

#[test]
fn test_model() {
    unsafe {
        let model = clp_sys::Clp_newModel();
        clp_sys::Clp_deleteModel(model);
    }
}
