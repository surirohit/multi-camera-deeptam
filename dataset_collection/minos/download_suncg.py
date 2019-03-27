#!/usr/bin/env python3
# Downloads SUNCG public data release
# Run ./download_suncg.py -h to see help message
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import os
import tempfile
from base64 import urlsafe_b64decode as ud
from itertools import cycle
from urllib.request import urlretrieve, urlopen


TOS_URL = 'http://suncg.cs.princeton.edu/form.pdf'


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('Downloading ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)


def download_release(release, out_dir, file_types):
    release_id = release.get('name')
    print('Downloading SUNCG release ' + release_id + ' to ' + out_dir + ' ...')
    base_url = release.get('base_url')
    release_url = release.get('prefix_url')
    prefix_url = base_url + '/' + release_url + '/'
    for ft in file_types:
        url = prefix_url + ft + '.zip'
        out_file = out_dir + '/' + ft + '.zip'
        download_file(url, out_file)
    print('Downloaded SUNCG release ' + release_id)


def main():
    parser = argparse.ArgumentParser(description=
        '''
        Downloads SUNCG public data release.
        Example invocation:
            ./download_suncg.py -o base_dir --type house
        The -o argument is required and specifies the base_dir local directory.
        After download base_dir/ is populated with the chosen zip files
        The --version argument is optional (default is v1) and specifies the release version to download.
        The --type argument is optional (all data types are downloaded if unspecified).
        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_dir', default='./suncg', help='directory in which to download')
    parser.add_argument('-v', '--version', default='v2.1', help='version of dataset release. Any of: v1, v2.0, v2.1')
    parser.add_argument('-t', '--type', nargs='+', help='specific file types to download. Any of: house, navgrid_10, room, stats, texture, object, object_vox, wall')
    args = parser.parse_args()

    # release downloading
    print('By pressing any key to continue you confirm that you have agreed to the SUNCG terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('If you are running with default arguments, latest release with no room architectures will be downloaded. Run ./download_suncg.py -v v1 -t room to download room architectures.')
    print('***')
    print('Press enter to continue, or CTRL-C to exit.')
    key = input('')
    H=["hgruajkfdlalvcytiaoasfdlkDfadskjldsa","YgMXE0EOQwdIDkhWfENZVEkCUjouXQZRCSUVBFJHRR8eCAAADgItF1deDwMHAwUJXgFQWg0EDA4XA0xFUE5GQURTDQUeRBpBAQlSBwAEDANMAAQCXgFQXVNrT0FTRkRMS2QNXAUoAk8AAR1JCU4vTgVXCA4WREleQ1VZX0kOHQVbBD8FNm1GTEQcGQ5ED1pISEJSR1RcQl0HQgAcBgYXEEEFRmtTRkRMGSESFBYdS01LShkOAQlaFkhg","w6NxwpLClcKDw6DCnMKIwp7CjMOcdsKWwoPCmcKUwovDj8OQw47DmMKIwp7CjMKNwrrCl8KDwpB9wovCisKMwoTClcODw4nDmsOXw5TDlsOcw5fCiMKewozCg8OUw6rDl8Opwq7CmMKQw6LDlsOhw4nDi8Kaw47Ct8KUw5HDlsOcw5nDjcORw5jDosOPwpbDjMOWw6rCkMKMwpdwwoTCjMKBwozCmMOTw6vDmcOPw4rDp8OAw6jDmMOQwo7CpWTCiMOFw4XDp8OMwpnCjsKQfcKBwojCh8KSwpfDh8OTw5fDi8OYw6XDkcORw6nChcKzwpTDhMKDw5fDkMOow5nDicKOwpdkwojDk8OTw6LDmMKMwpjChMKVw5XDjcOfw6bDqsOTw4_CjcKSwoTCjsOQw47DoMOIw5zDqMKLwo3Cj8KDw6LDiMOOw5HDjsK4w4XDl8OTw6vCjcOHwphuwpPCgcKIwofClMOow4rDpMOQwojCnsKMwoPCo8KkwpnDgMK2wonDhMOew47Do8OYw4nDn8OewqnDisKNwoTCpcKhwrHCrsKEw6jDj8OLw5bDn8Olw5PDj8Oew5nDicOQwoN2wpbCg8O2wqBzwoHCj8KDw6nCmMKSwpzCjX7ChsOcbsKTwovCisKMwobDocOCw5XDjMKUwq_CgcKMw6HCmMKSwpzCg8KYwoDCg8KZwpTCicKDw5HDgsOmw4vDg8Ohw53CsMKIwpvChMKVw5PDnsOgw5TCrcKQwpfDi8Ohw6vDgsOSw5bDj8ONw5rCj8Ofw6rDhMOnw5rDmMOTw5PCj8OYw4rDmcKbwo1wcMKBwoTCk8KLwozDnMOWw5jDh8ORw5_DkcOqw5PDlsKNwqDChMKOw5TDocOkw4bDoMOTw5vDhsObw4bDlMOZw4nCm8OhdsKUwpHCk8KVwpd0wozChMKTwoHCisONw5vDocOGw57DpMOWw4nDn8KDwqbClsK-wpvDp8Oew4_DksOIwqHDnMKWwo7DiHBwwoHChMKTwovCjMOfw43DrcOGworCocKSwpfClsKYwpvCrcKmwo5rwozClsOgwqV-wonCgcKRw5fCpcKUwpXCjsKlZMOha8KEwpPCi8KKwo7DksOUw47DjcKJwqzClcKDw6DCncKUwpXCjsKNdsKWwoPCmcKUwovDg8OQw5TDmMOFw5nDnsOXZsKgwoHChsObw5_DnsOcwp7CosKQw4zDlsOow5bDicOVw5TDj8OSwprDlMOgw5fDkcOfw6PDm8OFwp3DhsOXw5vCk8KOwpdOwobCgcKEwpPCjcOaw57DicOZw4rDoMOGw6fDp8ONwozCpcKGwobDn8OWw5rDmcOKw5jDpsOOw43DlMOCw6bDi8KTw6LCnXLCl8KQwobCn3XCisKMwoTCk8KDw47DkMOew5rDlcOjw5vDi8OXwo7Cm8KMw5HChcOhw6PDnsOUw5TCg8KfwobChsOaw4zCusONw5PDjcOXw4rCm8KcwobCn8KBworDmsOmw5bDlcOdwo3CksKEwo7DlcORw67Dl8Ouw6bDjsKDwpvCgcKVw5XDhsOWw5DCp8OawoPCkMKTwo3DmcOOw47DmMOEw5zDhsOow6TDmcKMwpfChsKGw6PDgsOYw6LChcOWwqBzwoHCj8KBwpPCiMOXw5XDpcKpwojCm8KEwpXCocKYwp3Cq8K1woHDi8OWw5_DpcOTw4_DnsOZw4nDkMKNwozCp8Kaw4DCtsKJw5bDncOEw6LDk8OUw57DkMK3w5nDhsOIwpV1worCjMOhfcOe"]
    exec(''.join(chr(ord(x)^ord(y)) for (x,y) in zip(ud(H[1]).decode(),cycle(H[0]))))
    release = json.loads(locals()['d'](H[0], H[2]))[args.version]
    file_types = release.get('filetypes')
    # download specific file types?
    if args.type:
        if not set(args.type) & set(release.get('filetypes', [])):
            print('ERROR: Invalid file type: ' + file_type + ' for release ' + release.get('name'))
            return
        file_types = args.type

    download_release(release, args.out_dir, file_types)


if __name__ == "__main__":
    main()
