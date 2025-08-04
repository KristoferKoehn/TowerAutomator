# TowerAutomator
an automation script to farm resources on The Tower mobile game

This project uses Android Debug Bridge with scrcpy to handle screen mirroring to desktop and touch input commands.

Screen detection is done with cv2 template searching and easyOCR Optical Character Recognition neural net for reading upgrade names/values/costs.

Requires android device and (so far) USB-C wired connection, probably 3.0+ data speeds.

Steps to operate:
load project in IDE (for now), import packages
On android device, turn on developer mode
set up USB debug on android device, turn on "always on" screen option
download scrcpy (https://github.com/Genymobile/scrcpy) and place extracted directory in /external_programs
open terminal in scrcpy directory, use this command: ./scrcpy --new-display --no-vd-system-decorations --start-app=com.TechTreeGames.TheTower --window-title="The Tower, Automator" --always-on-top
maximise the window, then press alt-w to get window to correct resolution
run the python project and start a round!


https://github.com/user-attachments/assets/5a0e8387-374f-4b74-bb38-f3c2de20f8fc

