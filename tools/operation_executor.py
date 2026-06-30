from tools.pandas_tool import PandasTool


class OperationExecutor:
    """
    Executes every operation returned
    by the Validation Agent 
    (In form of a JSON)
    """

    def __init__(self, dataframe):

        self.tool = PandasTool(dataframe)

    def execute(self, operations):

        for operation in operations:

            tool_name = operation["tool"]

            if tool_name == "drop_column":

                self.tool.drop_column(
                    operation["column"]
                )

            elif tool_name == "median_imputation":

                self.tool.median_imputation(
                    operation["column"]
                )

            elif tool_name == "remove_duplicates":

                self.tool.remove_duplicates()
            
            elif tool_name == "mode_imputation":

                self.tool.mode_imputation(
                    operation["column"]
                )

        return self.tool.get_dataframe()

    def get_logs(self):

        return self.tool.get_logs()

    def save_dataset(self, path):

        self.tool.save_dataset(path)