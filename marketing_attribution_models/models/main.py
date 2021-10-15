from my_model import HMODELS


def main(model_name, channels, conv_value):
    """
    Função que carrega um modelo que foi implementado
    e aplica em um pd.Series contendo listas de canais.

    Parameters
    ----------
    model_name : str or function
        Model name
    channels : pd.Series
        pd.Series contendo lista de canais
    conv_value : int or pd.Series
        If value is a pd.Series, obj must be the
        same lenght passed on 'channels' innit

    Returns
    -------
    results : pd.Series
    """
    my_model = HMODELS(channels)
    if callable(model_name):
        my_model.load_model(model_name)
    else:
        my_model.load_custom_model(model_name)
    my_model.fit()
    return my_model.apply_value(conv_value)
