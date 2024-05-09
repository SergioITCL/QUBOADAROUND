def get_activation_name(activation):

    if hasattr(activation, "__name__"):
        return activation.__name__.upper()
    
    return type(activation).__name__.upper()