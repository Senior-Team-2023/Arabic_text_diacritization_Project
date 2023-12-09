import json

class ConfigLoader:
    def __init__(self):
        self.data = {}

    def load_config(self):
        # Read the existing configuration from the JSON file
        with open('config.json', 'r') as config_file:
            self.data = json.load(config_file)
            print("Configurations Read Successfully")

            for key, value in self.data.items():
                self.data[key] = value

        return self.data
