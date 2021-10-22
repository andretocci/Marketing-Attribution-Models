import itertools
import math
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from .models import heuristic
from models.main import heuristic_models
import models.markov as mk
import data.random_data as random_data
import data_prep.journey as journey
import data_prep.group_data as group_data


# from .models import heuristic
# from .models.main import heuristic_models
# from .models import markov as mk
# from .data import random_data
# from .data_prep import journey
# from .data_prep import group_data


class MAM:
    """MAM (Marketing Attribution Models) is a class inspired on the R Package.

    ‘GameTheoryAllocation’ from Alejandro Saavedra-Nieves and ‘ChannelAttribution’ from
    Davide Altomare and David Loris that was created to bring these concepts to Python
    and to help us understand how the different marketing channels behave during the
    customer journey.

    Parameters:
    df = None by default, but should only be None if choosing to use a random dataframe.
         Otherwise, it has to receive a Pandas dataframe;
    time_till_conv_colname = None by default.
        Column name in the df containing the time in hours untill the
        moment of the conversion. The column must have the same elements as the
        channels_colname has. Values could be on a list ou a string with a separator;
        If your session is crashing here, try setting the variable
        time_till_conv_colname equal to 'skip_column'. But skipping this column you will
        not be able to run all the models in this class
    conversion_value = 1 by default.
        Integer that represents a monetary value of a 'conversion', can also receive a
        string indicating the column name on the dataframe containing the conversion
        values;
    channels_colname = None by default.
        Column name in the df containing the different channels during the customer
        journey. The column must have the same elements as the time_till_conv_colname
        has. Values could be on a list ou a string with a separator;
    journey_with_conv_colname = None by default.
        Column name in the df indicating if the journey (row) was a successfully
        conversion (True), or not (False).
    group_channels = False by default.
        Most important parameter on this class. This indicates the input format of the
        dataframe.
            True  = Each row represents a user session that will be grouped into a user
                    journey;
            False = Each row represents a user journey and the columns
    group_channels_by_id_list = Empty list by default.
    group_timestamp_colname = None by default.
    create_journey_id_based_on_conversion = False by default.
    path_separator = ' > ' by default.
        If using 'group_channels = True', this should match the separator being used on
        the inputed dataframe in the channels_colname;
    verbose = False by default.
        Internal parameter for printing while working with MAM;
    random_df = False by default.
        Will create a random dataframe with testing purpose;

    OBS: If your session is crashing, try setting the variable verbose True and some
    status and tips will be printed;
    """

    def __init__(
        self,
        df=None,
        time_till_conv_colname=None,
        conversion_value=1,
        channels_colname=None,
        journey_with_conv_colname=None,
        group_channels=False,
        group_channels_by_id_list=None,
        group_timestamp_colname=None,
        create_journey_id_based_on_conversion=False,
        path_separator=" > ",
        verbose=False,
        random_df=False,
    ):
        if not group_channels_by_id_list:
            group_channels_by_id_list = []

        self.verbose = verbose
        self.sep = path_separator
        self.group_by_channels_models = None
        # self.decode_channels = None

        if group_channels_by_id_list is None:
            group_channels_by_id_list = []

        ##########################################################
        ################## Instance attributes ###################
        ##########################################################

        self._first_click = None
        self._last_click = None
        self._last_click_non = None
        self._linear = None
        self._position_based = None
        self._time_decay = None
        self._position_decay = None

        #####################################################
        ##### Section 1: Creating object and attributes #####
        #####################################################

        ###########################
        #### random_df == True ####
        ###########################

        if random_df:
            df = random_data.data_frame()
            group_channels = True
            channels_colname = "channels"
            journey_with_conv_colname = "has_transaction"
            group_channels_by_id_list = ["user_id"]
            group_timestamp_colname = "visitStartTime"
            create_journey_id_based_on_conversion = True

        ################################
        #### group_channels == True ####
        ################################

        if group_channels:

            # Copying, sorting and converting variables
            df = (
                df.copy()
                .reset_index()
                .assign(timestamp=pd.to_datetime(df[group_timestamp_colname]))
                .sort_values(group_channels_by_id_list + ["timestamp"])
            )

            if create_journey_id_based_on_conversion:

                df = journey.journey_id_based_on_conversion(
                    df=df,
                    group_id=group_channels_by_id_list,
                    transaction_colname=journey_with_conv_colname,
                )
                group_channels_by_id_list = ["journey_id"]

            # Grouping channels based on group_channels_by_id_list
            ######################################################

            df_temp = group_data.group_channels(
                df,
                channels_colname,
                group_timestamp_colname,
                group_channels_by_id_list,
                print_log=False,
            )

            self.channels = df_temp["channel"].copy()
            self.time_till_conv = df_temp["time_till_conv"].copy()
            self._print("Status: Done")

            if journey_with_conv_colname is None:

                # If journey_with_conv_colname is None, we will assume that
                # all journeys ended in a conversion
                ###########################################################
                self.journey_with_conv = self.channels.apply(lambda x: True)
                self.journey_id = pd.Series(df[group_channels_by_id_list].unique())

            else:
                # Grouping unique journeys and whether the journey ended with a
                # conversion
                ##########################################################
                self._print("Grouping journey_id and journey_with_conv...")
                df_temp = df[group_channels_by_id_list + [journey_with_conv_colname]]
                temp_journey_id_conv = (
                    df_temp.groupby(group_channels_by_id_list)[
                        journey_with_conv_colname
                    ]
                    .max()
                    .reset_index()
                )
                self.journey_id = temp_journey_id_conv[group_channels_by_id_list]
                self._print("Status: Done")
                self.journey_with_conv = temp_journey_id_conv[journey_with_conv_colname]
                self._print("Status: Done")

            # conversion_value could be a single int value or a panda series
            if isinstance(conversion_value, int):
                self.conversion_value = self.journey_with_conv.apply(
                    lambda valor: conversion_value if valor else 0
                )
            else:
                self.conversion_value = (
                    df.groupby(group_channels_by_id_list)[conversion_value]
                    .sum()
                    .reset_index()[conversion_value]
                )

        #################################
        #### group_channels == False ####
        #################################
        else:
            # df = df.reset_index().copy()
            self.journey_id = df[group_channels_by_id_list]
            self._print("Status_journey_id: Done")

            #####################
            ### self.channels ###
            #####################

            # converts channels str to list of channels
            if isinstance(df[channels_colname][0], str):
                self._print("Status_journey_to_list: Working")
                self.channels = df[channels_colname].apply(lambda x: x.split(self.sep))
                self._print("Status_journey_to_list: Done")
            else:
                self.channels = df[channels_colname]
                self._print("Status_journey_to_list: Skipped")

            ###########################
            ### self.time_till_conv ###
            ###########################
            if time_till_conv_colname is None:
                self._print(
                    "If your session is crashing here, try setting the variable "
                    + "time_till_conv_colname equal to skip_column"
                )
                self.time_till_conv = self.channels.apply(
                    lambda x: list(range(len(x)))[::-1]
                )
                self.time_till_conv = self.time_till_conv.apply(
                    lambda x: list(np.asarray(x) * 24)
                )
            else:
                if time_till_conv_colname == "skip_column":
                    self.time_till_conv = None
                    print(
                        "Skipping this column you will not be able to run all the "
                        + "models in this class"
                    )
                else:
                    if isinstance(df[channels_colname][0], str):
                        self.time_till_conv = df[time_till_conv_colname].apply(
                            lambda x: [float(value) for value in x.split(self.sep)]
                        )
                    else:
                        self.time_till_conv = df[time_till_conv_colname]
            self._print("Status_time_till_conv: Done")

            ##############################
            ### self.journey_with_conv ###
            ##############################
            if journey_with_conv_colname is None:
                self.journey_with_conv = self.channels.apply(lambda x: True)
            else:
                self.journey_with_conv = df[journey_with_conv_colname]
            self._print("Status_journey_with_conv: Done")

            ########################
            ### conversion_value ###
            ########################

            # conversion_value could be a single int value or a panda series
            if isinstance(conversion_value, int):
                self.conversion_value = self.journey_with_conv.apply(
                    lambda valor: conversion_value if valor else 0
                )
            else:
                self.conversion_value = df[conversion_value]

        # if conversion_null_value is None:
        #   self.conversion_null_value = None
        # elif isinstance(conversion_value, str):
        #   self.conversion_null_value = df[conversion_null_value]

        #################
        ### DataFrame ###
        #################

        self.data_frame = None
        # self.get_dataframe()

    ######################################
    ##### Section 2: Output methods  #####
    ######################################

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def get_dataframe(self):
        """Return inputed attributes as a Pandas Data Frame on
        self.DataFrame."""
        if not isinstance(self.data_frame, pd.DataFrame):
            if isinstance(self.journey_id, pd.DataFrame):
                self.data_frame = self.journey_id.copy()
                self.data_frame["channels_agg"] = self.channels.apply(
                    self.sep.join
                ).copy()
                self.data_frame["converted_agg"] = self.journey_with_conv.copy()
                self.data_frame["conversion_value"] = self.conversion_value.copy()
            else:
                self.data_frame = pd.DataFrame(
                    {
                        "journey_id": self.journey_id.copy(),
                        "channels_agg": self.channels.apply(self.sep.join).copy(),
                        "converted_agg": self.journey_with_conv.copy(),
                        "conversion_value": self.conversion_value.copy(),
                    }
                )
            if self.time_till_conv is None:
                self.data_frame["time_till_conv_agg"] = None
            else:
                self.data_frame["time_till_conv_agg"] = self.time_till_conv.apply(
                    lambda x: self.sep.join([str(value) for value in x])
                ).copy()

        return self.data_frame

    def attribution_all_models(
        self,
        model_type="all",
        last_click_non_but_not_this_channel="Direct",
        time_decay_decay_over_time=0.5,
        time_decay_frequency=128,
        shapley_size=4,
        shapley_order=False,
        shapley_values_col="conv_rate",
        markov_transition_to_same_state=False,
        group_by_channels_models=True,
    ):
        """Runs all heuristic models on this class and returns a data frame.

        Models:
           - attribution_last_click_non
           - attribution_first_click
           - attribution_linear
           - attribution_position_based
           - attribution_time_decay
        Parameters:
        model_type = ['all',
                     'heuristic',
                     'algorithmic']
        """

        if model_type == "all":
            heuristic = True
            algorithmic = True
        elif model_type == "heuristic":
            heuristic = True
            algorithmic = False
        else:
            heuristic = False
            algorithmic = True

        if heuristic:
            # Running attribution_last_click
            self.attribution_last_click(
                group_by_channels_models=group_by_channels_models
            )

            # Running attribution_last_click_non
            self.attribution_last_click_non(
                but_not_this_channel=last_click_non_but_not_this_channel
            )

            # Running attribution_first_click
            self.attribution_first_click(
                group_by_channels_models=group_by_channels_models
            )

            # Running attribution_linear
            self.attribution_linear(group_by_channels_models=group_by_channels_models)

            # Running attribution_position_based
            self.attribution_position_based(
                group_by_channels_models=group_by_channels_models
            )

            # Running attribution_time_decay
            self.attribution_time_decay(
                decay_over_time=time_decay_decay_over_time,
                frequency=time_decay_frequency,
                group_by_channels_models=group_by_channels_models,
            )

        if algorithmic:

            # Running attribution_shapley
            self.attribution_shapley(
                size=shapley_size,
                order=shapley_order,
                group_by_channels_models=group_by_channels_models,
                values_col=shapley_values_col,
            )

            # Running attribution_shapley
            self.attribution_markov(
                transition_to_same_state=markov_transition_to_same_state
            )

        return self.group_by_channels_models

    def plot(
        self,
        *args,
        model_type="all",
        sort_model=None,
        number_of_channels=10,
        other_df=None,
        **kwargs,
    ):

        """Barplot of the results that were generated and stored on the
        variable self.group_by_channels_models.

        Parameters:
        model_type = ['all',
                       'heuristic'
                       'algorithmic']
        sort_model = has to be a string and accept regex by inputing r'example'
        other_df = None. In case the user wants to use a new data frame
        """

        model_types = {
            "all": "all",
            "heuristic": r"heuristic",
            "algorithmic": r"algorithmic",
        }

        if not isinstance(other_df, pd.DataFrame):
            # Checking if there are any results on self.group_by_channels_models
            if isinstance(self.group_by_channels_models, pd.DataFrame):
                df_plot = self.group_by_channels_models
            else:
                ax = "self.group_by_channels_models == None"
        else:
            df_plot = other_df

        # Sorting self.group_by_channels_models
        if sort_model is not None:
            # List comprehension to accept regex
            df_plot = df_plot.sort_values(
                [[x for x in df_plot.columns if re.search(sort_model, x)]][0],
                ascending=True,
            )

        # Selecting columns that matches the pattern
        if model_types[model_type] != "all":
            df_plot = df_plot[
                ["channels"]
                + [x for x in df_plot.columns if re.search(model_types[model_type], x)]
            ]

        # Subsetting the results based on the number of channels to be shown
        df_plot = df_plot.tail(number_of_channels)

        # Melting DF so the results are devided into 'channels', 'variable' and 'value'
        df_plot = pd.melt(df_plot, id_vars="channels")

        # Plot Parameters
        ax, _ = plt.subplots(figsize=(20, 7))
        ax = sns.barplot(
            data=df_plot, hue="variable", y="value", x="channels", *args, **kwargs
        )
        plt.xticks(rotation=15)
        ax.legend(loc="upper left", frameon=True, fancybox=True)
        ax.axhline(0, color="black", linestyle="-", alpha=1, lw=2)
        ax.grid(color="gray", linestyle=":", linewidth=1, axis="y")
        ax.set_frame_on(False)

        return ax

    def channels_journey_time_based_overwrite(
        self, selected_channel="Direct", time_window=24, order=1, inplace=False
    ):
        """Overwrites channels in the conversion jorney that matches the
        criteria with the previous channel in the journey:

          - Is equal to the selected_channel;
          - The diference between the contacts is less than the time_window selected;

        Parameters:
        selected_channel =
            Channel to be overwritten;
        time_window =
            The time window in hours that the selected channel will be overwritten;
        order =
            How many times the function will loop throught the same journey;
            ex: journey [Organic > Direct > Direct]
              order 1 output: [Organic > Organic > Direct]
              order 2 output: [Organic > Organic > Organic]
        """
        frame = self.channels.to_frame(name="channels")
        frame["time_till_conv_window"] = self.time_till_conv.apply(
            lambda time_till_conv: [time_window + 1]
            + [
                time - time_till_conv[i + 1]
                for i, time in enumerate(time_till_conv)
                if i < len(time_till_conv) - 1
            ]
        )
        frame["time_till_conv_window"] = frame["time_till_conv_window"].apply(
            lambda time_till_conv: np.absolute(np.asarray(time_till_conv)).tolist()
        )
        loop_count = 0
        while loop_count < order:
            frame["channels"] = frame.apply(
                lambda x: [
                    x.channels[i - 1]
                    if ((canal == selected_channel) & (time < time_window))
                    else canal
                    for i, (canal, time) in enumerate(
                        zip(x.channels, x.time_till_conv_window)
                    )
                ],
                axis=1,
            )
            loop_count += 1

        if inplace:
            self.channels = frame["channels"].copy()
            new_channels = None
        else:
            new_channels = frame["channels"].copy()

        return new_channels

    def __heuristic_model(self, model_name, args=None):
        """ """
        journey_res, grouped_res = heuristic_models(
            model_name,
            self.channels,
            self.conversion_value,
            self.journey_with_conv,
            self.time_till_conv,
            args=args,
        )
        model_name = f"attribution_{model_name}_heuristic"

        # Adding the results to self.DataFrame
        self.__save_journey_results(journey_res, model_name)

        # Results part 2: Results
        self.__save_group_by_results(grouped_res, model_name)

        return journey_res, grouped_res

    def __save_journey_results(self, journey_res, model_name):

        # Adding the results to self.DataFrame
        self.get_dataframe()
        self.data_frame[model_name] = (
            journey_res["result"]
            .apply(lambda x: self.sep.join([str(value) for value in x]))
            .copy()
        )

    def __save_group_by_results(self, grouped_res, model_name):
        """Internal function to generate the group_by_channels_models.

        A pandas DF containing the attributed values for each channel
        """
        # Grouped Results
        if isinstance(self.group_by_channels_models, pd.DataFrame):
            grouped_res = grouped_res.reset_index()
            grouped_res.columns = ["channels", model_name]
            self.group_by_channels_models = pd.merge(
                self.group_by_channels_models,
                grouped_res,
                how="outer",
                on=["channels"],
            ).fillna(0)
        else:
            self.group_by_channels_models = grouped_res.reset_index()
            self.group_by_channels_models.columns = ["channels", model_name]

        return grouped_res

    ##############################################
    #
    #
    #  Begin of new methods
    #
    #
    #################################

    def first_click_journeys(self):
        """Returns an object that contains First Click results with journey
        granularity."""
        if self._first_click is None:
            warnings.warn(
                "In order to call this method, attribution_first_click method must "
                + "be called first."
            )
        else:
            return self._first_click[0]

    def first_click_channels(self):
        """Returns an object that contains First Click results with channel
        granularity."""
        if self._first_click is None:
            warnings.warn(
                "In order to call this method, attribution_first_click method must "
                + "be called first."
            )
        else:
            return self._first_click[1]

    def last_click_journeys(self):
        """Returns an object that contains Last Click results with journey
        granularity."""
        if self._last_click is None:
            warnings.warn(
                "In order to call this method, attribution_last_click method must "
                + "be called first."
            )
        else:
            return self._last_click[0]

    def last_click_channels(self):
        """Returns an object that contains Last Click results with channel
        granularity."""
        if self._last_click is None:
            warnings.warn(
                "In order to call this method, attribution_last_click method must "
                + "be called first."
            )
        else:
            return self._last_click[1]

    def last_click_non_journeys(self):
        """Returns an object that contains Last Click ignoring a specific
        channel results with journey granularity."""
        if self._last_click_non is None:
            warnings.warn(
                "In order to call this method, attribution_last_click_non method "
                + "must be called first."
            )
        else:
            return self._last_click_non[0]

    def last_click_non_channels(self):
        """Returns an object that contains Last Click ignoring a specific
        channel results with channel granularity."""
        if self._last_click_non is None:
            warnings.warn(
                "In order to call this method, attribution_last_click_non method "
                + "must be called first."
            )
        else:
            return self._last_click_non[1]

    def linear_journeys(self):
        """Returns an object that contains Linear results with journey
        granularity."""
        if self._linear is None:
            warnings.warn(
                "In order to call this method, attribution_linear method must be "
                + "called first."
            )
        else:
            return self._linear[0]

    def linear_channels(self):
        """Returns an object that contains Linear results with channel
        granularity."""
        if self._linear is None:
            warnings.warn(
                "In order to call this method, attribution_linear method must be "
                + "called first."
            )
        else:
            return self._linear[1]

    def position_based_journeys(self):
        """Returns an object that contains Position based results with journey
        granularity."""
        if self._position_based is None:
            warnings.warn(
                "In order to call this method, attribution_position_based method "
                + "must be called first."
            )
        else:
            return self._position_based[0]

    def position_based_channels(self):
        """Returns an object that contains Position Based results with channel
        granularity."""
        if self._position_based is None:
            warnings.warn(
                "In order to call this method, attribution_position_based method "
                + "must be called first."
            )
        else:
            return self._position_based[1]

    def time_decay_journeys(self):
        """Returns an object that contains Time Decay results with journey
        granularity."""
        if self._time_decay is None:
            warnings.warn(
                "In order to call this method, attribution_time_decay method must "
                + "be called first."
            )
        else:
            return self._time_decay[0]

    def time_decay_channels(self):
        """Returns an object that contains Time Decay results with channel
        granularity."""
        if self._first_click is None:
            warnings.warn(
                "In order to call this method, attribution_time_decay method must "
                + "be called first"
            )
        else:
            return self._time_decay[1]

    ###################################################
    ##### Section 3: Channel Attribution methods  #####
    ###################################################

    def attribution_last_click(self, group_by_channels_models=True):
        """The last touchpoint receives all the credit.

        Parameters:
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models
        """
        model_name = "last_click"
        journey_res, grouped_res = self.__heuristic_model(model_name)
        self._last_click = (journey_res, grouped_res)

        return self._last_click

    def attribution_last_click_non(
        self, but_not_this_channel="Direct", group_by_channels_models=True
    ):
        """All the traffic from a Specific channel is ignored, and 100% of the credit
        for the sale goes to the last channel that the customer clicked through from
        before converting.

        Parameters:
        but_not_this_channel =
            Channel to be overwritten.
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models
        """
        model_name = "last_click_non"
        journey_res, grouped_res = self.__heuristic_model(
            model_name, (but_not_this_channel)
        )
        self._last_click_non = (journey_res, grouped_res)

        return self._last_click_non

    def attribution_first_click(self, group_by_channels_models=True):
        """The first touchpoint recieves all the credit.

        Parameters:
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models.
        """
        model_name = "first_click"
        journey_res, grouped_res = self.__heuristic_model(model_name)
        self._first_click = (journey_res, grouped_res)

        return self._first_click

    def attribution_linear(self, group_by_channels_models=True):
        """Each touchpoint in the conversion path has an equal value.

        Parameters:
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models.
        """
        model_name = "linear"
        journey_res, grouped_res = self.__heuristic_model(model_name)
        self._linear = (journey_res, grouped_res)

        return self._linear

    def attribution_position_based(
        self,
        list_positions_first_middle_last=None,
        group_by_channels_models=True,
    ):
        """First and last contact have preset values, middle touchpoints are evenly
        distributed with the chosen weight.

        default:
         - First channel = 0.4
         - Distributed among the middle channels = 0.2
         - Last channel = 0.4

        Parameters:
        list_positions_first_middle_last =
            List with percentages that will be given to each position
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models
        """

        model_name = "position_based"
        journey_res, grouped_res = self.__heuristic_model(
            model_name, (list_positions_first_middle_last)
        )
        self._position_based = (journey_res, grouped_res)

        return self._position_based

    def attribution_position_decay(self, group_by_channels_models=True):
        """Linear decay for each touchpoint further from conversion.

        Parameters:
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models

        OBS: This function is WIP.
        """
        model_name = "position_decay"
        journey_res, grouped_res = self.__heuristic_model(model_name)
        self._position_decay = (journey_res, grouped_res)

        return self._position_decay

    def attribution_time_decay(
        self, decay_over_time=0.5, frequency=168, group_by_channels_models=True
    ):
        """Decays for each touchpoint further from conversion.

        Parameters:
        decay_over_time =
            Percentage that will be lost by time away from the conversion.
        frequency =
            The frequency in hours that the decay will happen.
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models
        """

        if self.time_till_conv is None:
            print("time_till_conv is None, attribution_time_decay model will not work")
        else:
            model_name = "time_decay"
            journey_res, grouped_res = self.__heuristic_model(
                model_name, args=(decay_over_time, frequency)
            )
            self._time_decay = (journey_res, grouped_res)

        return self._time_decay

    def attribution_markov(
        self,
        transition_to_same_state=False,
        group_by_channels_models=True,
        conversion_value_as_frequency=True,
    ):
        """Attribution using Markov."""
        model_name = "attribution_markov"
        model_type = "_algorithmic"
        if transition_to_same_state:
            model_name = model_name + "_same_state" + model_type
        else:
            model_name = model_name + model_type

        temp = self.channels.apply(
            lambda x: ["(inicio)"] + x
        ) + self.journey_with_conv.apply(lambda x: ["(conversion)" if x else "(null)"])

        orig = []
        dest = []
        journey_length = []

        temp.apply(mk.save_orig_dest)

        # copying conversion_quantity to each new row
        if type(self.conversion_value) in (int, float):
            # we do not hava a frequency column yet so we are using
            # self.conversion_value.apply(lambda x: 1) to count each line
            conversion_quantity = self.conversion_value.apply(lambda x: 1)

        else:
            if conversion_value_as_frequency:
                freq_values = self.conversion_value
            else:
                freq_values = self.conversion_value.apply(lambda x: 1)

            conversion_quantity = []

            for a, b in zip(freq_values, journey_length):
                conversion_quantity.extend([a] * (b - 1))

        temp = pd.DataFrame({"orig": orig, "dest": dest, "count": conversion_quantity})
        temp = temp.groupby(["orig", "dest"], as_index=False).sum()
        self._print(temp)

        if not transition_to_same_state:
            temp = temp[temp.orig != temp.dest]

        # Converting channels_names to index and pass a numpy array foward
        channels_names = (
            ["(inicio)"]
            + list(
                (set(temp.orig) - set(["(inicio)"]))
                | (set(temp.dest) - set(["(conversion)", "(null)"]))
            )
            + ["(null)", "(conversion)"]
        )
        temp["orig"] = temp.orig.apply(channels_names.index)
        temp["dest"] = temp.dest.apply(channels_names.index)
        matrix = mk.path_to_matrix(temp[["orig", "dest", "count"]].values)
        removal_effect_result = mk.removal_effect(matrix)[1:-2]
        results = removal_effect_result / removal_effect_result.sum(axis=0)

        # Channels weights
        frame = pd.DataFrame({"value": results}, index=channels_names[1:-2])
        removal_effect_result = pd.DataFrame(
            {"removal_effect": removal_effect_result}, index=channels_names[1:-2]
        )

        # Transition matrix
        matrix = mk.normalize_rows(matrix)
        matrix = pd.DataFrame(matrix, columns=channels_names, index=channels_names)

        # Apply weights back to each journey
        chmap = {a: b[0] for a, b in zip(frame.index.values, frame.values)}
        channels_value = self.channels.apply(lambda y: [chmap[x] for x in y])
        channels_value = channels_value.apply(lambda x: list(np.array(x) / sum(x)))

        # Adding the results to self.DataFrame
        self.get_dataframe()
        self.data_frame[model_name] = channels_value.apply(
            lambda x: self.sep.join([str(value) for value in x])
        )

        # Grouping the attributed values for each channel
        total_conv_value = self.journey_with_conv * self.conversion_value
        if group_by_channels_models:
            if isinstance(self.group_by_channels_models, pd.DataFrame):
                frame = frame.reset_index()
                frame.columns = ["channels", model_name]
                frame[model_name] = frame[model_name] * total_conv_value.sum()
                self.group_by_channels_models = pd.merge(
                    self.group_by_channels_models, frame, how="outer", on=["channels"]
                ).fillna(0)
            else:
                frame = frame.reset_index()
                frame.columns = ["channels", model_name]
                frame[model_name] = frame[model_name] * total_conv_value.sum()
                self.group_by_channels_models = frame
        else:
            frame = "group_by_channels_models = False"

        return (channels_value, frame, matrix, removal_effect_result)

    def journey_conversion_table(self, order=False, size=None):
        """Transforms journey channels in boolean columns, count the number of
        conversions and journeys and compute the conversion rate of the channel
        combination.
        """
        # Creating Channels DF
        df_temp = self.journey_id.copy()

        if order:
            df_temp["combinations"] = self.channels.apply(
                lambda channels: sorted(list(set(channels)), key=channels.index)
            ).copy()
        else:
            df_temp["combinations"] = self.channels.apply(
                lambda channels: sorted(list(set(channels)))
            ).copy()

        if size is not None:
            df_temp["combinations"] = df_temp["combinations"].apply(
                lambda channels: self.sep.join(channels[size * -1 :])
            )
        else:
            df_temp["combinations"] = df_temp["combinations"].apply(self.sep.join)

        # Adding journey_with_conv column
        df_temp["journey_with_conv"] = self.journey_with_conv.apply(int)
        df_temp["conversion_value"] = self.conversion_value

        # Grouping journey_with_conv
        conv_val = (
            df_temp.groupby(["combinations"])["conversion_value"]
            .sum()
            .reset_index()["conversion_value"]
        )
        df_temp = (
            df_temp.groupby(["combinations"])["journey_with_conv"]
            .agg([("conversions", "sum"), ("total_sequences", "count")])
            .reset_index()
        )
        df_temp["conversion_value"] = conv_val
        # Calculating the conversion rate
        df_temp["conv_rate"] = df_temp["conversions"] / df_temp["total_sequences"]

        return df_temp

    def coalitions(self, size=4, unique_channels=None, order=False):
        """This function gives all the coalitions of different channels in a matrix.
        Most of the extra parameters are used when calculating Shapley's value with
        order.

        Parameters:
        size =
            Limits max size of unique channels in a single journey
        unique_channels =
            By default will check self.channels unique values, or a list of channels can
            be passed as well.
        order =
            Boolean that indicates if the order of channels matters during the process.
        """
        if unique_channels is None:
            unique_channels = list(set(sum(self.channels.values, [])))
        channels_combination = []

        # Creating a list with all the permutations if order is True
        if order is True:
            for L in range(0, size + 1):
                for subset in itertools.combinations(unique_channels, L):
                    channels_combination.append(list(subset))
        else:
            for L in range(0, size + 1):
                for subset in itertools.combinations(sorted(unique_channels), L):
                    channels_combination.append(list(subset))

        # Creating a DF with the channels as the boolean columns
        df_temp = pd.Series(channels_combination).to_frame(name="combinations")
        for channel in unique_channels:
            df_temp[channel] = df_temp.combinations.apply(
                lambda channels: any(channel in s for s in channels)
            )

        return df_temp

    def attribution_shapley(
        self,
        size=4,
        order=False,
        values_col="conv_rate",
        merge_custom_values=None,
        group_by_channels_models=True,
    ):
        """Defined by Wikipedia: The Shapley value is a solution concept in Cooperative
        Game Theory.

        It was named in honor of Lloyd Shapley, who introduced it in 1953. To each
        cooperative game it assigns a unique distribution (among the players) of a total
        surplus generated by the coalition of all players. Here in the context of
        marketing channels we can use the model to understand the valeu of the
        cooperation of channels to generate a conversion.

        Parameters:
        size =
            Limits max size of unique channels in a single journey. If there is a
            journey that has more channels than the defined limit, the last N channels
            will be considered. It's also important to accentuate that increasing the
            number of channels, increases the number calculations exponentially.
        order =
            Boolean that indicates if the order of channels matters during the process.
        values_col =
            The conversion rate is used by default, but the other columns in the
            journey_conversion_table can be used as well like 'conversions',
            'conversion_value'.
        merge_custom_values = None by defaut.
            Can be passed a Pandas Data Frame with two columns only, the first one
            representing the channels combination and the secong the custom value that
            you want to apply as the values_col. Will be merged(Left Join) with grouped
            self.journey_conversion_table() and applied a .fillna().
        group_by_channels_models = True by default.
            Will aggregate the attributed results by each channel on
            self.group_by_channels_models.
        """

        # Creating conv_table that will contain the aggregated results based on the journeys
        conv_table = self.journey_conversion_table(order=order, size=size)

        # Merge merge_custom_values
        if merge_custom_values is not None:
            if not isinstance(merge_custom_values, pd.DataFrame):
                print(
                    "Warning: variable merge_custom_values has to be a Pandas "
                    + "DataFrame containing two columns representing the channels "
                    + "combination and his conv value."
                )
            else:
                merge_custom_values.columns = ["combinations", "custom_value"]
                conv_table = pd.merge(
                    conv_table, merge_custom_values, on="combinations", how="left"
                ).fillna(0)
                values_col = "custom_value"

        # Removing all jouneys that have not converted
        conv_table = conv_table[conv_table.conversions > 0]
        channels_shapley = conv_table.combinations.apply(
            lambda x: x.split(self.sep)
        ).copy()
        results = []

        for journey in channels_shapley:
            n = len(journey)

            coalitions = self.coalitions(n, journey, order=order)
            coalitions.combinations = coalitions.combinations.apply(self.sep.join)
            coa = (
                coalitions[1:]
                .drop("combinations", axis=1)
                .astype(int)
                .astype(float)
                .reset_index(drop=True)
            )

            # Merging the coalitions table with the grouped results on conv_table
            valores = (
                pd.merge(coalitions, conv_table, on="combinations", how="left")[
                    values_col
                ]
                .fillna(0)
                .values
            )

            v = valores[1:]
            coaux = coa.copy()

            for line in list(range(0, ((2 ** n) - 1))):

                for channel in coa.columns:
                    s = len(coaux.iloc[line, :][coaux.iloc[line, :] != 0])
                    if coa[channel][line] == 0:
                        a = (
                            -(math.factorial(s) * math.factorial(n - s - 1))
                            / math.factorial(n)
                            * v[line]
                        )
                        coa[channel][line] = a
                    else:
                        b = (
                            (math.factorial(s - 1) * math.factorial(n - s))
                            / math.factorial(n)
                            * v[line]
                        )
                        coa[channel][line] = b

            results.append(list(coa.sum()))

        # Model col_name
        model_name = "attribution_shapley_size" + str(size) + "_" + values_col
        model_type = "_algorithmic"
        if order:
            model_name = model_name + "_order" + model_type
        else:
            model_name = model_name + model_type

        if (values_col == "conv_rate") or (values_col == "custom_value"):
            conv_table[model_name] = results
            conv_table[model_name] = (
                conv_table[model_name].apply(np.asarray) * conv_table["total_sequences"]
            )
            conv_table[model_name] = (
                conv_table[model_name].apply(lambda x: x / x.sum())
                * conv_table["conversion_value"]
            )
            conv_table[model_name] = conv_table[model_name].apply(lambda x: x.tolist())
        else:
            conv_table[model_name] = results

        ##########################
        # group_by_channels_models#
        ##########################

        # Aggregating the results by unique channel
        if group_by_channels_models:
            channels_list = sum(channels_shapley, [])
            values_list = sum(conv_table[model_name].values, [])
            frame = pd.DataFrame({"channels": channels_list, "value": values_list})
            frame = frame.groupby(["channels"])["value"].sum()

            if isinstance(self.group_by_channels_models, pd.DataFrame):
                frame = frame.reset_index()
                frame.columns = ["channels", model_name]
                self.group_by_channels_models = pd.merge(
                    self.group_by_channels_models, frame, how="outer", on=["channels"]
                ).fillna(0)
            else:
                self.group_by_channels_models = frame.reset_index()
                self.group_by_channels_models.columns = ["channels", model_name]
        else:
            frame = "group_by_channels_models=False"

        return (conv_table, frame)


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "channels": pd.Series([["x", "y", "z"], ["x", "y", "z", "y", "z"], ["z"]]),
            "conv_value": pd.Series([1, 7, 22]),
            "has_conv": pd.Series([True, True, False]),
            "time_values": pd.Series([[1680, 168, 0], [1680, 168, 55, 10, 0], [0]]),
        }
    )

    model = MAM(
        df=df,
        time_till_conv_colname="time_values",
        conversion_value="conv_value",
        channels_colname="channels",
        journey_with_conv_colname="has_conv",
        group_channels=False,
    )
    # model.attribution_all_models()
    model.attribution_first_click()
