# Master-Thesis
Knee joint modeling in Python
Program created for purposes of my Master thesis. Solves up to 10^7 configurations of knee displacement. 

# Setup (for Ubuntu)

Requires Python 3:

> sudo apt install python3

and python3 venv:

> sudo apt-get install -y python3-venv

Create virtual enviroment:

> python3 -m venv venv

Activate virtual enviroment:

> source venv/bin/activate

Install requirements.txt

> pip install -r requirements.txt

Run program:

> ./venv/bin/python "./Knee workspace_main program.py"

Please be patient it should take up to 30s to solve workspace.
You can rotate and translate knee however you want. Green cloud inside the joint indicates area where tibia coordinate system can be moved without damage.


