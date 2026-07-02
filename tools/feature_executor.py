from tools.feature_engineering_tool import FeatureEngineeringTool


class FeatureExecutor:

    """
    Executes feature engineering operations
    returned by the Feature Validation Agent.
    """

    def __init__(self, dataframe):

        self.tool = FeatureEngineeringTool(dataframe)

    # =====================================================

    def execute(self, operations):

        for operation in operations:

            tool_name = operation["tool"]

            if tool_name == "one_hot_encode":

                self.tool.one_hot_encode(
                    operation["column"]
                )

            elif tool_name == "label_encode":

                self.tool.label_encode(
                    operation["column"]
                )

            elif tool_name == "standard_scale":

                self.tool.standard_scale(
                    operation["column"]
                )

            elif tool_name == "minmax_scale":

                self.tool.minmax_scale(
                    operation["column"]
                )

            elif tool_name == "robust_scale":

                self.tool.robust_scale(
                    operation["column"]
                )

            elif tool_name == "log_transform":

                self.tool.log_transform(
                    operation["column"]
                )
            else:

                print(
                    f"Unknown Feature Engineering Tool : {tool_name}"
                )

        return self.tool.get_dataframe()

    # =====================================================

    def get_logs(self):

        return self.tool.get_logs()
    

    def save_dataset(self, path):

        self.tool.save_dataset(path)
        
    def get_fitted_objects(self):

        return self.tool.get_fitted_objects()