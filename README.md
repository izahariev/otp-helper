# OTP Helper

OTP Helper is a tool for dealing with players who only play a single hero aka "One Trick Ponies (OTP)" in Heroes of the
Storm game. With the help of this tool you can identify OTPs and ban their mains in the hero select phase of the game.

## Configuration

Install Tesseract OSR on your system

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

You will have to manually update the "otp.py" script "scan" function with the coordinates of the enemy players names on
your screen. The default configured "names_coordinates" are for a screen of resolution 1920:1080. You will have to also
create "db" folder and "images" folder in the project directory.

## Functionality

The OTP Helper provides the following functionality:

- scan - Check for OTPs in the enemy team and show info for their mains
- add - Add a player to the database
- update - Update a player mains
- replace - Replace a player mains
- list - List all players in the data base
- count - Count of players in the data base

For additional info and usage use the help option of the tool:

```bash
python otp.py --help
```

or for help with a specific functionality use "-h" flag with the selected option:

```bash
python otp.py add -h
```

```bash
python otp.py update -h
```