#!/usr/bin/env python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
radiorec.py – Recording internet radio streams
Copyright (C) 2013  Martin Brodbeck <martin@brodbeck-online.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import argparse
import configparser
import datetime
import os
import stat
import sys
import threading
import urllib.request
import time

# instantiate
from numba import jit

config = configparser.ConfigParser()

def check_duration(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Duration must be a positive integer.')

    if value < 1:
        raise argparse.ArgumentTypeError(
            'Duration must be a positive integer.')
    else:
        return value


def read_settings():
    settings_base_dir = ''
    if sys.platform.startswith('linux'):
        settings_base_dir = ''
    elif sys.platform == 'win32':
        settings_base_dir = os.getenv('LOCALAPPDATA') + os.sep + 'radiorec'
    elif sys.platform == 'darwin':
        settings_base_dir = os.getenv(
            'HOME') + os.sep + 'Library' + os.sep + 'Application Support' + os.sep + 'radiorec'
    settings_base_dir += os.sep
    config = configparser.ConfigParser()
    try:
        config.read('radiorec_settings.ini')
    except FileNotFoundError as err:
        print(str(err))
        print('Please copy/create the settings file to/in the appropriate '
              'location.')
        sys.exit()
    return config


def my_record_worker(stoprec, station, target_dir, linux_public, fileName, duration):
    settings = read_settings()
    url = settings.get('STATIONS', station)
    #cur_dt_string = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    conn = urllib.request.urlopen(url)
    filename = target_dir + os.sep + fileName
    content_type = conn.getheader('Content-Type')
    if content_type == 'audio/mpeg':
        filename += '.mp3'
    elif content_type == 'application/ogg' or content_type == 'audio/ogg':
        filename += '.ogg'
    elif content_type == 'audio/x-mpegurl':
        print('Sorry, M3U playlists are currently not supported')
        sys.exit()
    else:
        print('Unknown content type "' + content_type + '". Assuming mp3.')
        filename += '.mp3'

    with open(filename, "wb") as target:
        if linux_public:
            print('Apply public write permissions (Linux only)')
            os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                     stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)

        # Assuming a 128kbps stream, 16000 bytes are one second of music
        length = int(16000 * duration)
        target.write(conn.read(length))


def my_record(url, duration, file_name):
    streamurl = url
    if streamurl.endswith('.m3u'):
        with urllib.request.urlopen(streamurl) as remotefile:
            for line in remotefile:
                if not line.decode('utf-8').startswith('#') and len(line) > 1:
                    tmpstr = line.decode('utf-8')
                    break
        streamurl = tmpstr
    target_dir = "data/test"
    stoprec = threading.Event()

    recthread = threading.Thread(target=my_record_worker, args=(stoprec, streamurl, target_dir, False, file_name, duration))
    recthread.setDaemon(True)
    recthread.start()
    recthread.join(timeout=duration)

    if recthread.is_alive:
        stoprec.set()
        recthread.join()


def list(args):
    settings = read_settings()
    for key in sorted(settings['STATIONS']):
        print(key)