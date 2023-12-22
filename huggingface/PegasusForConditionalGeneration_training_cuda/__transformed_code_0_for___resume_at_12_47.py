def __helper_outer_function():
    # this is a helper function to help compilers generate bytecode to read capture variables from closures, rather than reading values from global scope. The value of these variables does not matter, and will be determined in runtime.
    __class__ = None
    def __transformed_code_0_for___resume_at_12_47(___stack0, self, key, value):
        nonlocal __class__
        import importlib
        
        
        def __resume_at_16_48(___stack0):
            nonlocal __class__
            return None
        
        
        return __resume_at_16_48(importlib.import_module('builtins').super(
            __class__, self).__setattr__(key, value))
