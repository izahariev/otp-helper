# OTP Helper

OTP Helper is a tool for dealing with players who only play a single hero aka "One-Trick Ponies (OTP)" in Heroes of the
Storm game.
With the help of this tool, you can identify OTPs and ban their mains in the hero select phase of the game.

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
- replace - Replace player mains
- replace player name image - Replace player image in the database
- show player name image - Show player name image in the database
- list - List all players in the database
- count - Count of players in the database

For additional info and usage, use the help option of the tool:

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

## How to use

### Scan

The "scan" function will search for the players in the enemy team in the database, and if found, will print info about
their mains. When your team is banning run the "scan" option as shown in the example bellow:

```bash
python otp.py scan
```

The chat should be closed and your mouse should not be on any of the
enemy players names.

***IMPORTANT: It is possible that the tool sometimes mistakes a player name with another name. If this happens ignore
the output. It is possible to reduce or remove entirely the wrong suggestions, however in doing so, due to the fact that
the same player in a different position on the screen has slightly different colors, we will miss some players for which
we have data. The approach taken is to make sure we always recognize players for which we have data with the
downside of sometimes getting wrong names. Missing an OTP may result in losing a game while reading a wrong name is a
slight inconvenience.***

The scan will also make a screenshot of the names of the enemy players, which can be used with the add function to add
new entries to the database. The images will be placed in the "images" folder in the project folder, and will be named
from 1 to 5 representing the position of the corresponding player on the screen starting from the top.

### Add

To add a player, a "scan" must have been performed before that.
Call the function by passing the player index (position in the enemy team starting from 1 and from the top),
the name of the player (supports all UTF-8 symbols) and the player mains in the following format:
{priority}-{hero name}, where priority is between P0 and P5.
You can pass any string for
priority if you want to use anything else.
Bellow is an example of how to use the "add" function:

```bash
python otp.py add 1 d0m0v0y "P1-Chromie"
```

If the player already exists, an error message will be displayed and nothing will be added to the database.

### Update

The "update" function will add the specified heroes info to the player in the database. The syntax is the same as when
adding a new entry:

```bash
python otp.py update 1 d0m0v0y "P1-Jaina"
```

If the player does not exist, an error message will be displayed.

### Replace

The "replace" functionality will replace the specified player heroes info in the database. This means the old heroes
info will be replaced with the given one. The parameters are the same as the "add" and "update" functions:

```bash
python otp.py replace 1 d0m0v0y "P1-Thrall"
```

### Replace player name image

The "replace player name image" functionality will replace the image of the given player with the image with the given
index created by the last scan

### Show player name image

The "show player name image" functionality will show the given player name.
The image window can be closed by pressing any key

### List

The list functionality will print all available players names in the database in the console:

```bash
python otp.py list
```

### Count

The count functionality will print the number of entries in the database in the console:

```bash
python otp.py count
```