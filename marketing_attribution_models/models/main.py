import pandas as pd

from my_model import HMODELS


def heuristic_models(
    model_name, channels, conv_value=None, has_conv=None, values=None, args=None
):
    """
    Função que carrega um modelo que foi implementado
    e aplica em um pd.Series contendo listas de canais.

    Parameters
    ----------
    model_name : str or function
        Model name
    channels : pd.Series
        pd.Series containing a list
    conv_value : int or pd.Series
        If value is a pd.Series, obj must be the
        same lenght passed on 'channels' innit
    has_conv : pd.Series
        If value is a pd.Series, obj must be the
        same lenght passed on 'channels' innit
    values : pd.Series
        Series containing a lists of values to be used
        if needed for the model, like the time decay function

    Returns
    -------
    results : pd.Series
    """

    # Creating the objetc
    my_model = HMODELS(channels, values)

    # Loading model
    ## Checking if a custom function was passed
    if callable(model_name):
        try:
            my_model.load_custom_model(model_name)
        except Exception as e:
            print("Custom function must be applyied to a list and retur a list\n", e)

    else:
        my_model.load_model(model_name)

    # Fitting the model
    if args is None:
        my_model.fit()
    else:
        my_model.fit(*args)

    # Appling value to the results
    if conv_value is not None:
        my_model.apply_value(conv_value)

    # Appling value to the results
    if has_conv is not None:
        my_model.apply_value(has_conv)

    return my_model.get_df(), my_model.group_results()


if __name__ == "__main__":
    channels = pd.Series([["x", "y", "z"], ["x", "y", "z", "y", "z"], ["z"]])
    conv_value = pd.Series([1, 7, 22])
    has_conv = pd.Series([True, True, False])

    model = "last_click_non"
    custom_param = "z"
    results = heuristic_models(model, channels, conv_value, has_conv, args=custom_param)
    print(results)

    model = "last_click"
    custom_param = None
    results = heuristic_models(model, channels, conv_value, has_conv, args=custom_param)
    print(results)

    model = "time_decay"
    custom_param = None
    time_values = pd.Series([[1680, 168, 0], [1680, 168, 55, 10, 0], [0]])
    results = heuristic_models(
        model, channels, conv_value, has_conv, time_values, args=custom_param
    )
    print(results)
