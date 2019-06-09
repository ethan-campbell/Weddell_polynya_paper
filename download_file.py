# -*- coding: utf-8 -*-

from numpy import *
import os
import shutil
import requests
import subprocess
import gzip
import time
from ftplib import FTP
from datetime import datetime


def single_file(url_dir,filename,save_to,ftp_root=False,overwrite=False,verbose=True,auth=None):
    """ Downloads and saves a file from a given URL.

    Notes:
        - For HTTP downloads, if '404 file not found' error returned, function will return without
          downloading anything.
        - For FTP downloads, if given filename doesn't exist in directory, function will return without
          downloading anything.
    
    Args:
        url_dir: URL up to the filename, including ending slash
            NOTE: for ftp servers, include URL after the root, without starting slash
        filename: filename, including suffix
        save_to: directory path
        ftp_root: root URL of ftp server without preamble (ftp://) or ending slash, or 'False' if using HTTP
    
    """
    starting_dir = os.getcwd()
    
    try:
        if starting_dir is not save_to:
            os.chdir(save_to)
        if filename in os.listdir():
            if not overwrite:
                if verbose: print('>>> File ' + filename + ' already exists. Leaving current version.')
                return
            else:
                if verbose: print('>>> File ' + filename + ' already exists. Overwriting with new version.')

        if not ftp_root:
            full_url = url_dir + filename

            def get_func(url, stream=True, auth_key=None):
                try:
                    return requests.get(url, stream=stream, auth=auth_key)
                except requests.exceptions.ConnectionError as error_tag:
                    # note: this solution is super hacky and bad practice,
                    #       see https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
                    time.sleep(1)
                    return get_func(url, stream=stream, auth_key=auth_key)

            response = get_func(full_url, stream=True, auth_key=auth)

            if response.status_code == 404:
                if verbose: print('>>> File ' + filename + ' returned 404 error during download.')
                return
            with open(filename,'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
        else:
            ftp = FTP(ftp_root)
            ftp.login();
            ftp.cwd(url_dir);
            contents = ftp.nlst();
            if filename in contents:
                local_file = open(filename,'wb')
                ftp.retrbinary('RETR ' + filename, local_file.write);
                local_file.close()
            ftp.quit();
    finally:
        os.chdir(starting_dir)


def all_files(ftp_root,url_dir,save_to,overwrite=False,verbose=True):
    """ Uses df.single_file() to download all files found in an FTP directory.
    """
    ftp = FTP(ftp_root)
    ftp.login();
    ftp.cwd(url_dir);
    contents = ftp.nlst();
    ftp.quit();
    for filename in contents:
        single_file(url_dir,filename,save_to,ftp_root,overwrite,verbose)


def un_gzip(dir,filename,append_extension='',remove_compressed_file=False,overwrite=False):
    """ Uncompresses a GZIP (.gz) file, saves to the same directory, and deletes the original.

    Only acts if compressed file exists and uncompressed file doesn't exist yet.

    """
    starting_dir = os.getcwd()

    try:
        if starting_dir is not dir:
            os.chdir(dir)
        new_filename = os.path.splitext(filename)[0] + append_extension
        if filename in os.listdir():
            if new_filename not in os.listdir() or overwrite:
                if overwrite and new_filename in os.listdir():
                    os.remove(dir + '/' + new_filename)
                inF = gzip.open(filename, 'rb')
                outF = open(new_filename, 'wb')
                outF.write(inF.read())
                inF.close()
                outF.close()
                if remove_compressed_file:
                    os.remove(filename)
    finally:
        os.chdir(starting_dir)


def convert_to_hdf5(script_dir, filename, old_data_dir, new_data_dir, overwrite=False, delete_original=False):
    """ Converts a single file from HDF4 to HDF5 format using a command line utility from internet.

    Converts given .hdf file in old_data_dir. Stores new HDF5 file as a copy in new_data_dir.
    If given HDF4 file does not exist, function will return without doing anything.

    Args:
        script_dir: directory path of h4toh5 executable (command line) script
        old_data_dir: directory path of original HDF4 data file
        new_data_dir: directory path for new, converted HDF5 data file
        NOTE: all directory paths should contain trailing slash ('/')

    """

    starting_dir = os.getcwd()
    os.chdir(new_data_dir)
    new_data_files = os.listdir()
    os.chdir(old_data_dir)
    old_data_files = os.listdir()
    os.chdir(script_dir)

    new_filename = filename.split('.')[0] + '.h5'
    if filename in old_data_files:
        if (new_filename not in new_data_files) or (new_filename in new_data_files and overwrite is True):
            command = './h4toh5 "' + old_data_dir + filename + '"'
            subprocess.call(command,shell=True)
            if old_data_dir is not new_data_dir:
                os.rename(old_data_dir + new_filename, new_data_dir + new_filename)
        if delete_original:
            os.remove(old_data_dir + filename)
    os.chdir(starting_dir)


def how_far(index, all_vals, interval):
    """ Prints percent-completion notices while iterating through a list.

    Args:
        index: current index within all_vals
        all_vals: a list
        interval: e.g. 0.1 = print notices at 10%-completion intervals
    """
    percents_comp = floor(linspace(0, 100, 1 / interval + 1))
    percents_comp_indices = floor(linspace(0, len(all_vals), 1 / interval + 1))
    if index in percents_comp_indices:
        percent_comp = percents_comp[int(where(percents_comp_indices == index)[0])]
        print('>>> ' + str(datetime.now()) + ' - action is ' + str(percent_comp) + '% complete')
