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


def record_worker(stoprec, streamurl, target_dir, args):
    conn = urllib.request.urlopen(streamurl)
    cur_dt_string = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    filename = target_dir + os.sep + args.station
    if args.name:
        filename += '_' + args.name
    content_type = conn.getheader('Content-Type')
    if (content_type == 'audio/mpeg'):
        filename += '.mp3'
    elif (content_type == 'application/ogg' or content_type == 'audio/ogg'):
        filename += '.ogg'
    elif (content_type == 'audio/x-mpegurl'):
        print('Sorry, M3U playlists are currently not supported')
        sys.exit()
    else:
        print('Unknown content type "' + content_type + '". Assuming mp3.')
        filename += '.mp3'

    with open(filename, "wb") as target:
        if args.public:
            verboseprint('Apply public write permissions (Linux only)')
            os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                     stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        verboseprint('Recording ' + args.station + '...')
        while not stoprec.is_set() and not conn.closed:
            target.write(conn.read(1024))


def my_record_worker(stoprec, station, target_dir, linux_public, fileName, duration, args):
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
            verboseprint('Apply public write permissions (Linux only)')
            os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                     stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)

        # Assuming a 128kbps stream, 16000 bytes are one second of music
        target.write(conn.read(16000 * int(duration)))
        # while not conn.closed:
        #     if time.time() > timeout:
        #         break
        #     target.write(conn.read(1024))


def record(args):
    settings = read_settings()
    streamurl = ''
    global verboseprint
    verboseprint = print if args.verbose else lambda *a, **k: None

    try:
        streamurl = settings['STATIONS'][args.station]
    except KeyError:
        print('Unkown station name: ' + args.station)
        sys.exit()
    if streamurl.endswith('.m3u'):
        verboseprint('Seems to be an M3U playlist. Trying to parse...')
        with urllib.request.urlopen(streamurl) as remotefile:
            for line in remotefile:
                if not line.decode('utf-8').startswith('#') and len(line) > 1:
                    tmpstr = line.decode('utf-8')
                    break
        streamurl = tmpstr
    verboseprint('stream url: ' + streamurl)
    target_dir = os.path.expandvars(settings['GLOBAL']['target_dir'])
    stoprec = threading.Event()

    recthread = threading.Thread(target=record_worker,
                                 args=(stoprec, streamurl, target_dir, args))
    recthread.setDaemon(True)
    recthread.start()
    recthread.join(args.duration * 60)

    if (recthread.is_alive):
        stoprec.set()


def my_record(url, duration, file_name, args=None):

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

    recthread = threading.Thread(target=my_record_worker, args=(stoprec, streamurl, target_dir, False, file_name, duration, args))
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


def main():
    parser = argparse.ArgumentParser(description='This program records '
                                                 'internet radio streams. It is free '
                                                 'software and comes with ABSOLUTELY NO '
                                                 'WARRANTY.')
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_record = subparsers.add_parser('record', help='Record a station')
    parser_record.add_argument('station', type=str,
                               help='Name of the radio station '
                                    '(see `radiorec.py list`)')
    parser_record.add_argument('duration', type=check_duration,
                               help='Recording time in minutes')
    parser_record.add_argument('name', nargs='?', type=str,
                               help='A name for the recording')
    parser_record.add_argument(
        '-p', '--public', action='store_true',
        help="Public write permissions (Linux only)")
    parser_record.add_argument(
        '-v', '--verbose', action='store_true', help="Verbose output")
    parser_record.set_defaults(func=record)
    parser_list = subparsers.add_parser('list', help='List all known stations')
    parser_list.set_defaults(func=list)

    if not len(sys.argv) > 1:
        print('Error: No argument specified.\n')
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
