extern crate clp_sys;

pub struct Model {
    p_model: *mut clp_sys::Clp_Simplex,
}

impl Model {
    pub fn new() -> Model {
        let model = unsafe {
            clp_sys::Clp_newModel()
        };
        Model {
            p_model: model
        }
    }

    pub fn resize(&mut self, rows: usize, cols: usize) {
        unsafe {
            clp_sys::Clp_resize(self.p_model, rows as i32, cols as i32);
        }
    }

    pub fn number_rows(&self) -> usize {
        unsafe {
            clp_sys::Clp_numberRows(self.p_model) as usize
        }
    }

    pub fn number_columns(&self) -> usize {
        unsafe {
            clp_sys::Clp_numberColumns(self.p_model) as usize
        }
    }

    pub fn add_rows(&mut self, number: usize, row_lower: Vec<f64>, row_upper: Vec<f64>, row_starts: Vec<u32>, columns: Vec<u32>, elements: Vec<f64>) {
        unsafe {
            clp_sys::Clp_addRows(self.p_model, number as i32, row_lower.as_ptr(), row_upper.as_ptr(), row_starts.as_ptr() as *const i32, columns.as_ptr() as *const i32, elements.as_ptr());
        }
    }

    pub fn delete_rows(&mut self, number: usize, which: Vec<u32>) {
        unsafe {
            clp_sys::Clp_deleteRows(self.p_model, number as i32, which.as_ptr() as *const i32);
        }
    }

    pub fn add_columns(&mut self, number: usize, column_lower: Vec<f64>, column_upper: Vec<f64>, objective: Vec<f64>, column_starts: Vec<u32>, rows: Vec<u32>, elements: Vec<f64>) {
        unsafe {
            clp_sys::Clp_addColumns(self.p_model, number as i32, column_lower.as_ptr(), column_upper.as_ptr(), objective.as_ptr(), column_starts.as_ptr() as *const i32, rows.as_ptr() as *const i32, elements.as_ptr());
        }
    }

    pub fn delete_columns(&mut self, number: usize, which: Vec<u32>) {
        unsafe {
            clp_sys::Clp_deleteColumns(self.p_model, number as i32, which.as_ptr() as *const i32);
        }
    }

    pub fn change_row_lower(&mut self, row_lower: Vec<f64>) {
        unsafe {
            clp_sys::Clp_chgRowLower(self.p_model, row_lower.as_ptr());
        }
    }

    pub fn change_row_upper(&mut self, row_upper: Vec<f64>) {
        unsafe {
            clp_sys::Clp_chgRowUpper(self.p_model, row_upper.as_ptr());
        }
    }

    pub fn change_column_lower(&mut self, column_lower: Vec<f64>) {
        unsafe {
            clp_sys::Clp_chgColumnLower(self.p_model, column_lower.as_ptr());
        }
    }

    pub fn change_column_upper(&mut self, column_upper: Vec<f64>) {
        unsafe {
            clp_sys::Clp_chgColumnUpper(self.p_model, column_upper.as_ptr());
        }
    }

    pub fn change_objective_coefficients(&mut self, obj_in: Vec<f64>) {
        unsafe {
            clp_sys::Clp_chgObjCoefficients(self.p_model, obj_in.as_ptr());
        }
    }

    pub fn primal_tolerance(&self) -> f64 {
        unsafe {
            clp_sys::Clp_primalTolerance(self.p_model)
        }
    }

    pub fn set_primal_tolerance(&mut self, value: f64) {
        unsafe {
            clp_sys::Clp_setPrimalTolerance(self.p_model, value);
        }
    }

    pub fn dual_tolerance(&self) -> f64 {
        unsafe {
            clp_sys::Clp_dualTolerance(self.p_model)
        }
    }

    pub fn set_dual_torelance(&mut self, value: f64) {
        unsafe {
            clp_sys::Clp_setDualTolerance(self.p_model, value);
        }
    }

    pub fn dual_objective_limit(&self) -> f64 {
        unsafe {
            clp_sys::Clp_dualObjectiveLimit(self.p_model)
        }
    }

    pub fn set_dual_objective_limit(&mut self, value: f64) {
        unsafe {
            clp_sys::Clp_setDualObjectiveLimit(self.p_model, value);
        }
    }

    pub fn objective_offset(&self) -> f64 {
        unsafe {
            clp_sys::Clp_objectiveOffset(self.p_model)
        }
    }

    pub fn set_objective_offset(&mut self, value: f64) {
        unsafe {
            clp_sys::Clp_setObjectiveOffset(self.p_model, value);
        }
    }

    pub fn number_iterations(&self) -> usize {
        unsafe {
            clp_sys::Clp_numberIterations(self.p_model) as usize
        }
    }

    pub fn set_number_iterations(&mut self, value: usize) {
        unsafe {
            clp_sys::Clp_setNumberIterations(self.p_model, value as i32);
        }
    }

    pub fn maximum_iterations(&self) -> usize {
        unsafe {
            clp_sys::maximumIterations(self.p_model) as usize
        }
    }

    pub fn set_maximum_iterations(&mut self, value: usize) {
        unsafe {
            clp_sys::Clp_setMaximumIterations(self.p_model, value as i32);
        }
    }

    pub fn maximum_seconds(&self) -> f64 {
        unsafe {
            clp_sys::Clp_maximumSeconds(self.p_model)
        }
    }

    pub fn set_maximum_seconds(&mut self, value: f64) {
        unsafe {
            clp_sys::Clp_setMaximumSeconds(self.p_model, value);
        }
    }

    pub fn hit_maximum_iterations(&self) -> usize {
        unsafe {
            clp_sys::Clp_hitMaximumIterations(self.p_model) as usize
        }
    }

    pub fn problem_status(&self) -> usize {
        unsafe {
            clp_sys::Clp_status(self.p_model) as usize
        }
    }

    pub fn set_problem_status(&mut self, status: usize) {
        unsafe {
            clp_sys::Clp_setProblemStatus(self.p_model, status as i32);
        }
    }

    pub fn secondary_status(&self) -> usize {
        unsafe {
            clp_sys::Clp_status(self.p_model) as usize
        }
    }

    pub fn set_secondary_status(&mut self, status: usize) {
        unsafe {
            clp_sys::Clp_setProblemStatus(self.p_model, status as i32);
        }
    }

    pub fn optimization_direction(&self) -> f64 {
        unsafe {
            clp_sys::Clp_optimizationDirection(self.p_model)
        }
    }

    pub fn set_optimization_direction(&mut self, value: f64) {
        unsafe {
            clp_sys::Clp_setOptimizationDirection(self.p_model, value);
        }
    }

    pub fn primal_row_solution(&self) -> Vec<f64> {
        let n = self.number_rows();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_primalRowSolution(self.p_model), n, n)
        }
    }

    pub fn primal_column_solution(&mut self) -> Vec<f64> {
        let n = self.number_columns();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_primalColumnSolution(self.p_model), n, n)
        }
    }

    pub fn dual_row_solution(&self) -> Vec<f64> {
        let n = self.number_columns();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_dualRowSolution(self.p_model), n, n)
        }
    }

    pub fn dual_column_solution(&mut self) -> Vec<f64> {
        let n = self.number_rows();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_dualColumnSolution(self.p_model), n, n)
        }
    }

    pub fn row_lower(&self) -> Vec<f64> {
        let n = self.number_rows();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_rowLower(self.p_model), n, n)
        }
    }

    pub fn row_upper(&self) -> Vec<f64> {
        let n = self.number_rows();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_rowUpper(self.p_model), n, n)
        }
    }

    pub fn objective(&self) -> Vec<f64> {
        let n = self.number_rows();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_objective(self.p_model), n, n)
        }
    }

    pub fn column_lower(&self) -> Vec<f64> {
        let n = self.number_columns();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_columnLower(self.p_model), n, n)
        }
    }

    pub fn column_upper(&self) -> Vec<f64> {
        let n = self.number_columns();
        unsafe {
            Vec::from_raw_parts(clp_sys::Clp_columnUpper(self.p_model), n, n)
        }
    }

    pub fn objective_value(&self) -> f64 {
        unsafe {
            clp_sys::Clp_objectiveValue(self.p_model)
        }
    }

    pub fn primal(&mut self) -> i32 {
        unsafe {
            clp_sys::Clp_primal(self.p_model, 0)
        }
    }

    pub fn log_level(&self) -> usize {
        unsafe {
            clp_sys::Clp_logLevel(self.p_model) as usize
        }
    }

    pub fn set_log_level(&mut self, value: usize) {
        unsafe {
            clp_sys::Clp_setLogLevel(self.p_model, value as i32);
        }
    }
}
