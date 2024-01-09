import subprocess
from unittest.mock import patch

def run_script(script_path, input_values):
    with patch('builtins.input', side_effect=input_values):
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        print(result.stdout)

if __name__ == '__main__':
    script_path = '/queryVector.py'  # Replace with the path to your Python script
    games = [
    "Action Adventure Games",
    "n",
    "RPG Games for PlayStation",
    "The Guy Game",
    "Spyder Man",
    "star wars battlefront",
    "Iron Man",
    "James Earl Cash",
    "Crazy Taxi",
    "James Bond",
    "The Lord of the Rings The Two Towers PS2",
    "n"
]

    run_script(script_path, games)