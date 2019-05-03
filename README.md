**A realtime speech-music descriminator for the Raspbery Pi**

**Requirements**

LLVM:

On the _Raspian_ platform install LLVM version _3.8.0_ (http://releases.llvm.org/download.html#3.8.0). On Ubuntu 18.10/19.04 version 8.0.0 (`http://releases.llvm.org/download.html#8.0.0`) is recommended.

**Installation**

The software uses _pip_ and _virtual environments_ for managing packages.

Manual installation steps:

1. Install _ffmpeg_: `sudo apt install ffmpeg`
2. Clone repository: `git clone https://github.com/maxgraf96/bachelorarbeit.git && cd bachelorarbeit`
3. Install pip: `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py`
4. Install virtualenv: `sudo pip3 install virtualenv`
5. Create virtualenv: `virtualenv venv`
6. Activate virtualenv: `source venv/bin/activate`
7. Install required packages: `pip3 install -r requirements_desktop.txt` OR `pip3 install -r requirements_raspberry.txt`, depending on your system.
8. Execute `python3 Run.py oe1 10s` to test if everything works correctly.

_Note_:
Use piwheels (https://www.piwheels.org/) for installing packages on the Raspberri Pi! PiWheels provides prebuild packages. This significantly reduces installation time.

**Tested environments**

This software was tested with Python 3.5.2 (Raspberry Pi/Ubuntu Mate Xenial) and 3.6.7 (Dell XPS 15/Ubuntu 18.10).
