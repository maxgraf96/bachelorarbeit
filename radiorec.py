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

import configparser
import os
import stat
import sys
import threading
import urllib.request

def read_settings():
    """
    Reads the configuration (available radio stations and their URLs) from the "radiorec_settings.ini" config file
    :return: The configuration as a dict
    """
    settings_base_dir = ''
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

def record_worker(station, target_dir, linux_public, fileName, duration):
    """
    The worker that performs the actual streaming (in a separate thread)
    :param station: The station URL from which to stream from
    :param target_dir: The target directory in which the stream is saved
    :param linux_public: Apply write permissions if on linux
    :param fileName: The name of the saved file
    :param duration: The duration of the stream (in seconds)
    """
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

        # Assuming a 192kbps stream, 24000 bytes are one second of music
        length = int(24000 * duration)
        target.write(conn.read(length))


def record(url, duration, file_name):
    """
    Record from the given url for a given duration and store mp3 in given path
    :param url: The url from which to stream from. Must be an mp3 streaming source
    :param duration: The duration for which the data should be streamed in seconds
    :param file_name: The name of the file in which the data should be stored
    """
    streamurl = url
    if streamurl.endswith('.m3u'):
        with urllib.request.urlopen(streamurl) as remotefile:
            for line in remotefile:
                if not line.decode('utf-8').startswith('#') and len(line) > 1:
                    tmpstr = line.decode('utf-8')
                    break
        streamurl = tmpstr
    target_dir = "data/test"

    recthread = threading.Thread(target=record_worker, args=(streamurl, target_dir, False, file_name, duration))
    recthread.setDaemon(True)
    recthread.start()
    recthread.join(timeout=duration)