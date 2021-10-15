import warnings
from numpy.lib.function_base import diff

import pandas as pd

import heuristic as ht


class HMODELS:
    def __init__(self, channels):
        self.model = None
        self.results = None
        self.df = None
        self.channels = channels

    def load_model(self, model_name):
        """
        Loads a model from heuristic.py based on
        the function name

        Parameters
        ----------
        model_name : str
            Model name

        Returns
        -------
        None
        """
        self.model = getattr(ht, model_name)

    def load_custom_model(self, model):
        """
        Loads a custom model.

        Parameters
        ----------
        model : function
            Model function

        Returns
        -------
        None
        """
        if callable(model):
            warnings.warn(
                "In order to call this method, model param must " + "be a function"
            )
        else:
            self.model = model

    def __get_channels(self):
        """
        Loads self.channels attribute.

        Returns
        -------
        self.channels : pd.Series
        """
        return self.channels

    def __get_model(self):
        """
        Loads self.model attribute.

        Returns
        -------
        self.model : function
        """
        if self.model is None:
            warnings.warn(
                "In order to call this method, load_model method must "
                + "be called first"
            )
        else:
            return self.model

    def fit(self, args=None):
        """
        Apply loaded model.

        Returns
        -------
        None
        """
        if args is None:
            self.results = self.channels.apply(lambda x: self.__get_model()(x))
        else:
            self.results = self.channels.apply(lambda x: self.__get_model()(x, *args))

    def __get_results(self):
        """
        Loads self.results attribute.

        Returns
        -------
        self.results : pd.Series
        """
        if self.results is None:
            warnings.warn(
                "In order to call this method, fit method must " + "be called first"
            )
        else:
            return self.results

    def apply_value(self, value):
        """
        results * value

        Parameters
        ----------
        value : int or pd.Series
            If value is a pd.Series, obj must be the
            same lenght passed on 'channels' innit

        Returns
        -------
        results : pd.Series
        """
        self.results = self.__get_results() * value

    def generate_df(self):
        """ """
        self.df = pd.DataFrame(
            {"channel": self.__get_channels(), "result": self.__get_results()}
        )

    def get_df(self):
        """ """
        if self.df is None:
            self.generate_df()

        return self.df

    def group_results(self):
        """ """
        return (
            pd.DataFrame(
                {
                    "channel": self.__get_channels().explode(),
                    "result": self.__get_results().explode(),
                }
            )
            .query("result > 0")
            .groupby(["channel"])["result"]
            .sum()
        )


if __name__ == "__main__":
    channels = pd.Series([["x", "y", "z"], ["x", "y", "z", "y", "z"], ["z"]])
    values = pd.Series([1, 7, 22])
    my_model = HMODELS(channels)
    my_model.load_model("last_click_non")
    my_model.fit("z")
    my_model.apply_value(values)
    print(my_model.get_df(), "\n", my_model.group_results())
