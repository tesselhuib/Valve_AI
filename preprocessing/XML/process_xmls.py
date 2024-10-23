import os
import sys
import re
import warnings
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from tqdm import tqdm
from lxml import etree
from scipy import signal
from typing import (
    List, Self, Dict, Optional, Any, Literal,
)

# To find ecgprocess, can also install ecgprocess as package, then remove this
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ecgProcess.constants import (ProcessDicomNames as PDNames)
from ecgProcess.errors import (
    is_type, NotCalledError, STDOUT_MSG, Error_MSG, _check_readable, _check_presence
    )
from ecgProcess.utils.general import assign_empty_default
from plot_ecgs import ECGDrawing


class ECGXMLReader():

    __slots__ = (PDNames.LEAD_VOLTAGES, PDNames.LEAD_VOLTAGES2,
                 PDNames.RESULTS_DICT, PDNames.SOP_UID, PDNames.MEDIAN_PRESENT)
    
    resample_500: bool = True

    def __init__(self, lead_voltages=None, lead_voltages2=None,
                 results_dict={}, ecg_id=None):
        '''
        Initialises slots entries to `None`.
        '''
        setattr(self, PDNames.LEAD_VOLTAGES, lead_voltages)
        setattr(self, PDNames.LEAD_VOLTAGES2, lead_voltages2)
        setattr(self, PDNames.RESULTS_DICT, results_dict)
        setattr(self, PDNames.SOP_UID, ecg_id)
        setattr(self, PDNames.MEDIAN_PRESENT, True)

    def _resampling_500hz(self, frequency: int | float):
        """
        Re-sample the frequency to 500 hz.

        Parameters
        ----------
        frequency : int or float
            The original sampling frequency. Will simply check that this is
            not 500 and if not proceed with the resampling.

        """
        if int(frequency) != 500:
            # #### wave forms
            lead_volt_temp = getattr(self, PDNames.LEAD_VOLTAGES)
            for i in lead_volt_temp:
                lead_volt_temp[f"{i}"] =\
                    signal.resample(lead_volt_temp[f"{i}"], 5000)
            setattr(self, PDNames.LEAD_VOLTAGES, lead_volt_temp)
            # #### median beats
            if getattr(self, PDNames.MEDIAN_PRESENT) is True:
                lead_volt_temp2 = getattr(self, PDNames.LEAD_VOLTAGES2)
                for i in lead_volt_temp2:
                    lead_volt_temp2[f"{i}"] = \
                        signal.resample(lead_volt_temp2[f"{i}"], 600)
                setattr(self, PDNames.LEAD_VOLTAGES2, lead_volt_temp2)

    def __call__(self, path: str) -> Self:
        '''
        Parse xml and create dictionary with leads and their values.
        Also update Total number of samples and lead units for use in
        plotting the ECG.

        Parameters
        ----------
        path (str): path to xml file

        Returns
        -------
        Updated class
        Waveforms (dict): dictionary with leads as key and their voltages
            arranged in an array as values
        GeneralInfo (dict): dictionary with updated metadata on lead units
            and total number of samples

        '''
        # Create empty instance of results_dict to store all intermediate updates
        results_dict = {}
        # Create parser and parse xml file
        parser = etree.XMLParser(ns_clean=True, huge_tree=True)
        xml_f = etree.parse(path, parser)

        waveform_dict = {}
        median_waveform_dict = {}

        # Find StripData element with WaveformData and info on waveforms.
        strip_data = xml_f.find('.//StripData')

        sampling_number_value = int((strip_data.find('ChannelSampleCountTotal')).text)
        results_dict[PDNames.SAMPLING_NUMBER] = sampling_number_value

        sf = int((strip_data.find('SampleRate')).text)
        results_dict[PDNames.SF] = sf

        resolution_element = strip_data.find('Resolution')
        resolution_value = float(resolution_element.text)
        resolution_units = resolution_element.get('units', 'unkown')

        # Create conversion factor for voltage units
        if resolution_units == 'uVperLsb':
            conversion_factor = resolution_value
            results_dict[PDNames.LEAD_UNITS] = PDNames.MICROVOLT
        elif resolution_units == 'mVperLsb':
            conversion_factor = resolution_value
            results_dict[PDNames.LEAD_UNITS] = PDNames.MILLIVOLT
        else:
            print('Units resolution not uVperLsb or mVperLsb, but', resolution_units, '. Please check.')

        # Create dictionary with Leads as keys and Waveform arrays as values.
        for waveform in strip_data.findall('WaveformData'):
            lead = waveform.get('lead')
            data_text = waveform.text.strip()
            data_array = np.array(data_text.split(','), dtype=float)
            data_array_microvolts = data_array * conversion_factor
            data_array_microvolts_zero_mean = data_array_microvolts - np.average(data_array_microvolts)
            waveform_dict[lead] = data_array_microvolts_zero_mean
        # Do the same, but now for median waves

        median_sample_element = xml_f.find('.//MedianSamples')

        try:
            m_sampling_number_value = int((median_sample_element.find('ChannelSampleCountTotal')).text)
            results_dict[PDNames.SAMPLING_NUMBER_M] = m_sampling_number_value

            for m_waveform in median_sample_element.findall('WaveformData'):
                m_lead = m_waveform.get('lead')
                m_data_text = m_waveform.text.strip()
                m_data_array = np.array(m_data_text.split(','), dtype=float)
                m_data_array_microvolts = m_data_array * conversion_factor
                median_waveform_dict[m_lead] = m_data_array_microvolts

            setattr(self, PDNames.MEDIAN_PRESENT, True)
        
        except:
            setattr(self, PDNames.MEDIAN_PRESENT, False)

        if self.resample_500 is True:
            # the function will internally adjust
            # LEAD_VOLTAGES, and LEAD_VOLTAGES2
            # Nothing is returned
            self._resampling_500hz(frequency=results_dict[PDNames.SF])
            results_dict[PDNames.OVERSAMPLED] = True
            results_dict[PDNames.SF] = 500
            results_dict[PDNames.SAMPLING_FREQ] = 500
        else:
            results_dict[PDNames.OVERSAMPLED] = False

        # Set class attributes
        setattr(self, PDNames.LEAD_VOLTAGES, waveform_dict)
        setattr(self, PDNames.SOP_UID, os.path.splitext(path)[0])
        getattr(self, PDNames.RESULTS_DICT).update(results_dict)

        if getattr(self, PDNames.MEDIAN_PRESENT) is True:
            setattr(self, PDNames.LEAD_VOLTAGES2, median_waveform_dict)

        # Return updated class
        return self


class ECGXMLTable(object):
    '''
    Takes an `ECGXMLReader` instance and loops over a list of xml paths
    and maps these to which can be used in analyses or saved to disk.

    Attributes
    ----------
    RawPathList : list [`str`]
        A list of xml file paths.

    Methods
    -------
    get_table(update_keys,**kwargs)
        extracts data from multiple xml files and maps these to
        pandas.DataFrames.
    write_ecg(target_tar, target_path, sep, mode, compression, update_keys,
    **kwargs)
        writes each xml file to a single sets of files (metadata,
        waveforms, median beats). This is done by appending the extracted
        data from each file the target file set, minimising the memory
        footprint.
    write_pdf(ecgdrawing, target_path, write_failed, kwargs_reader,
    kwargs_drawing, kwargs_savefig)
        writes xml files to pdfs using the xml file name as file name.
    write_numpy(ecgdrawing, target_path, write_failed, kwargs_reader,
    kwargs_drawing, kwargs_savefig)
        writes xml files to npzs using the xml file name as file name.
    '''

    def __init__(self, ecgxmlreader: ECGXMLReader, path_list: List[str],
                 info_type: Literal['all', 'rhythm', 'median', 'meta'] = 'all'
                 ) -> None:
        """
        Initialises a new instance of `ECGXMLTable`.

        Parameters
        ----------
        ecgxmlreader : ECGXMLReader
            An instance of the ECGXMLReader data class.
        path_list : list [`str`]
            A list of paths to one or more .dcm files.
        """
        EXP_INFO = [PDNames.INFO_TYPE_ALL, PDNames.INFO_TYPE_RTM,
                    PDNames.INFO_TYPE_MED, PDNames.INFO_TYPE_MET,
                    ]
        self.INFO_MET = [PDNames.INFO_TYPE_ALL, PDNames.INFO_TYPE_MET]
        self.INFO_MED = [PDNames.INFO_TYPE_ALL, PDNames.INFO_TYPE_MED]
        self.INFO_RTM = [PDNames.INFO_TYPE_ALL, PDNames.INFO_TYPE_RTM]

        # #### check input
        is_type(ecgxmlreader, ECGXMLReader, 'ecgxmlreader')
        is_type(path_list, list, 'path_list')
        if info_type not in EXP_INFO:
            raise ValueError(f'`info_type` is restricted to `{EXP_INFO}`.')
        self.ecgxmlreader = ecgxmlreader
        setattr(self, PDNames.INFO_TYPE, info_type)
        setattr(self, PDNames.RPATH_L, path_list)

    def __str__(self):
        CLASS_NAME = type(self).__name__
        return (f"{CLASS_NAME} instance with "
                f"ecgxmlreader={self.ecgxmlreader}, "
                f"path_list={getattr(self, PDNames.RPATH_L)}."
                )

    def __repr__(self):
        CLASS_NAME = type(self).__name__
        return (f"{CLASS_NAME}(ecgxmlreader={self.ecgxmlreader}, "
                f"augment_leads={getattr(self, PDNames.RPATH_L)})"
                )

    def __call__(self,
                 skip_missing: Literal['Permissions', 'Data', 'None'] = 'Permissions',
                 verbose: bool = False,
                 ) -> Self:
        """
        Will take a ECGXMLReader and loops over a list of dcm file paths and
        and confirm the files exist and have appropriate read permission

        Parameters
        ----------
        skip_missing : {'Permissions', 'Data', 'None'}, default `Permissions`
            Whether files with a `PermissionError` or `AttributeError` should
            be skipped, or whether all errors should be raised.
            A PermissionError implies either the file is absent or it cannot
            be read. An AttributeError indicates the waveform_array
            attribute is absent from the pydicom instance.
            `Permissions` will only skip PermissionErrors, while `Data` will
            skip both PermssionErrors and AttributeErrors (due to a missing
            waveform_array), `None` will not skip errors.
        verbose : bool, default `False`
            Prints missing files if skip_missing is set to `True`.

        Attributes
        ----------
        FailedPathList : list [`str`]
            File paths which were either absent or without read permission.
        CuratedPathList : list [`str`]
            File paths which are readable.

        Returns
        -------
        self : `ECGXMLTable` instance
            Returns the class instance with updated attributes.
        """
        is_type(skip_missing, str)
        is_type(verbose, bool)
        self.skip_missing = skip_missing
        self.verbose = verbose
        SKIP_MISSING = [PDNames.SKIP_PERMISSIONS,
                        PDNames.SKIP_DATA,
                        PDNames.SKIP_NONE,
                        ]
        if skip_missing not in SKIP_MISSING:
            raise ValueError(Error_MSG.CHOICE_PARM.\
                             format('skip_missing', ', '.join(SKIP_MISSING)))
        # #### loop over path
        empty_list = []
        curated_list = []
        # #### loop over individual xml files and assign to self
        for p in getattr(self, PDNames.RPATH_L):
            try:
                # add p and remove if skipp_missing == True
                curated_list.append(p)
                _check_readable(p)
            except PermissionError as PE:
                empty_list.append(p)
                # try next
                if self.skip_missing != PDNames.SKIP_NONE:
                    curated_list.pop()
                    pass
                else:
                    raise PE
        setattr(self, PDNames.FPATH_L, empty_list)
        setattr(self, PDNames.CPATH_L, curated_list)
        # #### do we want to print empty_list
        if self.verbose is True:
            if len(getattr(self, PDNames.FPATH_L))>0:
                warnings.warn(
                    'The following files could not be accessed or found: {}.'.\
                    format(getattr(self, PDNames.FPATH_L))
                )
        # #### Return
        return self

    def _get_long_table(self, lead_list: List[Dict[str, np.ndarray]],
                        wave_type: str,
                        update_keys: Optional[Dict[str, str]] = None,
                        purge_header: bool = True,
                        **kwargs,
                        ) -> pd.DataFrame:
        '''
        Mapping lists of dictionaries containing lead specific numpy arrays
        to a long-formatted pandas table.

        Parameters
        ----------
        lead_list : list [`dict`]
            A list of dictionaries with the lead mapped to the keys and the
            voltage mapped to the values.
        wave_type : str,
            Adds a column `Waveform type` containing this string as a constant.
        update_keys : dict [`str`, `str`], default `NoneType`
            A dictionary to remap lead names: [`old`, `new`]
        purge_header : bool, default True
            Set to `False` to make sure the file header persists between calls.
            This is used to ensure the headers are the same between files.
        **kwargs : optional
            keyword arguments passed to pd.DataFrame.

        Attributes
        ----------
        kwargs_pdd : optional
            The keyword arguments supplied to pd.DataFrame.

        Returns
        -------
        self : pd.DataFrame
            A long-formatted table with lead, voltage, sampling indicator
            columns grouped by file (including an unique file indicator column)
        '''
        # #### check input and set constants
        is_type(lead_list, list, 'lead_list')
        is_type(wave_type, str, 'wave_type')
        self.kwargs_pdd = kwargs
        PREV = 'previous'
        # do we need to purge the header information
        # otherwise this persists throughout calls
        if purge_header is True:
            setattr(self, PREV, None)
        else:
            if not hasattr(self, PREV):
                setattr(self, PREV, None)
        # #### initiate table and map lists
        table = pd.DataFrame()
        for w, k in zip(lead_list, getattr(self, PDNames.KEY_L)):
            # do we need to remap key names
            if update_keys is not None:
                w = {update_keys.get(k, k): v for k, v in w.items()}
            # confirm the dictionary keys are identical
            current_keys = list(w.keys())
            if getattr(self, PREV) is not None:
                if current_keys != getattr(self, PREV):
                    raise KeyError('The dictionaries contain distinct keys. '
                                   'The last set of valid keys was {}, '
                                   'compared to {}.'.\
                                   format(getattr(self, PREV), current_keys))
            # if the same map to dataframe
            temp_df = pd.DataFrame(w, **self.kwargs_pdd)
            temp_df[PDNames.SOP_UID] = k
            table = pd.concat([table, temp_df], ignore_index=True)
            # clean
            setattr(self, PREV, current_keys)
            del current_keys
        # #### formatting to long
        table[PDNames.SAMPLING_SEQ] = table.groupby(
            PDNames.SOP_UID).cumcount()
        long_table = pd.melt(table, id_vars=\
                             [PDNames.SOP_UID, PDNames.SAMPLING_SEQ],
                             value_vars=getattr(self, PREV),
                             var_name=PDNames.COL_LEAD,
                             value_name=PDNames.COL_VOLTAGE
                             )
        long_table[PDNames.COL_WAVETYPE] = wave_type
        # #### return
        return long_table
    
    def write_pdf(self, ecgdrawing: ECGDrawing,
                  target_path: str = '.', write_failed: bool = True,
                  kwargs_reader: Dict[Any, Any] | None = None,
                  kwargs_drawing: Dict[Any, Any] | None = None,
                  kwargs_savefig: Dict[Any, Any] | None = None,
                  ) -> Self:
        '''
        Extracts xml files, and write these one by one to a pdf files
        using a supplied `ECGDrawing` instance.

        Parameters
        ----------
        ecgdrawing : ECGDrawing
            An instance of the ECGDrawing data class.
        target_path : `str`, default '.'
            The full path where the pdfs should be written to.
        write_failed : `bool`, default `True`
            Whether to write a text file to disk containing the failed file
            names.
        kwargs_*: dict [`any`, `any`], default `NoneType`
            dictionaries with keyword arguments for the `plt.savefig`,
            ECGDrawing, or ECGXMLReader instances.

        Attributes
        ----------
        target_path : `str`
            The directory or tar file path were the files are written to.
        NoDataList : `list` [`str`]
            A list of xml files without a waveform_array.

        Returns
        -------
        self : `ECGXMLTable` instance
            Returns the class instance with updated attributes.

        Notes
        -----
        The xml UID instance will be used as file name for the pdfs.

        Raises
        ------
        NotADirectoryError or PermissionError
            If the target directory does not exist or is not writable.
        '''
        # #### check input and set constants
        is_type(kwargs_reader, (type(None), dict))
        is_type(kwargs_drawing, (type(None), dict))
        is_type(kwargs_savefig, (type(None), dict))
        is_type(ecgdrawing, ECGDrawing)
        is_type(target_path, (pathlib.PosixPath, str))
        is_type(write_failed, bool)
        # check readability
        _check_presence(target_path)
        _check_readable(target_path)
        # map None to dict
        kwargs_savefig, kwargs_reader, kwargs_drawing = assign_empty_default(
                [kwargs_savefig, kwargs_reader, kwargs_drawing], dict)
        # map info type to type of plot
        if getattr(self, PDNames.INFO_TYPE) == PDNames.INFO_TYPE_MET:
            raise AttributeError('`info` should be {0} or {1}.'.format(
                PDNames.INFO_TYPE_MED, PDNames.INFO_TYPE_RTM))
        elif getattr(self, PDNames.INFO_TYPE) == PDNames.INFO_TYPE_ALL:
            signal_type = PDNames.WAVETYPE_RHYTHM
        else:
            signal_type = getattr(self, PDNames.INFO_TYPE)
        # #### create target path
        if target_path == '.':
            target_path = os.getcwd()
        # assign to self
        target = target_path
        print(target)
        setattr(self, PDNames.WRITE_ECG_PATH, target)
        # #### check if __call__ has been run
        if not hasattr(self, PDNames.CPATH_L):
            raise NotCalledError()
        # #### extract xml data
        key_list, no_data_list = [[] for _ in range(2)]
        # loop over individual xml files
        for p in tqdm(getattr(self, PDNames.CPATH_L), desc="Processing", unit="file"):
            no_data_list, key_list, ecg_inst = self._write_internal(
                path=p, no_data_list=no_data_list, key_list=key_list,
                **kwargs_reader,
                )
            if isinstance(ecg_inst, str):
                continue

            artist = ecgdrawing(ecgreader=ecg_inst, wave_type=signal_type,
                                **kwargs_drawing)

            # Check if median waveform is present, if not, skip PDF creation
            if getattr(ecg_inst, PDNames.MEDIAN_PRESENT) is False:
                print(f"The file {re.sub(r"[ ,\-\(\)\{\}]", '_', os.path.basename(key_list[-1]))} does not contain a median waveform. ECG recording probably failed. Skipping PDF creation.")
                continue

            # Check if motion artefacts are present, if so, skip PDF creation
            if not artist.check_all_ecg_leads_threshold():
                print(f"One or more ECG leads exceed the threshold. Skipping PDF creation for {re.sub(r"[ ,\-\(\)\{\}]", '_', os.path.basename(key_list[-1]))}.")
                continue

            filename_pdf = re.sub(
                r"[ ,\-\(\)\{\}]", '_', os.path.basename(key_list[-1])) + '.pdf'
            plt.savefig(
                fname=os.path.join(target, filename_pdf), **kwargs_savefig
                )
            plt.close(artist.fig)
        # #### write failed files, note not compressing these
        DELIM = '\t'
        if write_failed is True:
            # adding the reason for failing
            total_failures = [
                (p, PDNames.SKIP_PERMISSIONS) for p in
                getattr(self, PDNames.FPATH_L)] + [
                (p, PDNames.SKIP_DATA) for p in
                    getattr(self, PDNames.FAILED_DATA_L)]
            # writing to text file
            with open(os.path.join(target, PDNames.FAILED_FILE), 'w') as file:
                for p, cause in total_failures:
                    file.write(p + DELIM + cause + "\n")
        # #### return
        return self

    def _write_internal(self, path: str, no_data_list: list[str],
                        key_list: list[str],
                        **kwargs,
                        ) -> tuple[list, list, ECGXMLReader | str]:
        '''
        An internal function to read ECG xml data, check whether there these
        data may have been extracted before (compared to key_list) and record
        whether some files did not contain waveform_array attributes.

        Parameters
        ----------
        no_data_list : list [`str`]
            A list of file names without an waveform_array attribute.
        key_list : list [`str`]
            A list of xml UIDs which were processed before.

        Returns
        -------
        `tuple`
            A tuple with `no_data_list`, `key_list`, and an ECGXMLReader
            instance.

        Raises
        ------
        AttributeError
            raised if waveform_array or SOPinstanceUID attributes are absent.
        IndexError
            raised if a xml with the same SOPinstanceUID is processed
        '''

        if self.verbose is True:
            print(STDOUT_MSG.PROCESSING_PATH.format(path), file=sys.stdout)
        # get instance
        try:
            ecg_inst = self.ecgxmlreader(path, **kwargs)
        except AttributeError as AE:
            if self.skip_missing == PDNames.SKIP_DATA:
                no_data_list.append(path)
                # moving to the next path - cannot use continue here
                return no_data_list, key_list, 'continue'
            else:
                raise AE
        # extract unique identifier and check if it has been used before
        if hasattr(ecg_inst, PDNames.SOP_UID) is False:
            raise AttributeError(Error_MSG.MISSING_ATTR.format(
                PDNames.SOP_UID, 'ecg_inst'))
        key = str(getattr(ecg_inst, PDNames.SOP_UID))
        if key in key_list:
            raise IndexError('{0}:{1} was already extracted before. Please '
                             'ensure the supplied files are unique.'.
                             format(PDNames.SOP_UID, key))
        else:
            key_list.append(key)
        # extract the remaining
        setattr(self, PDNames.FAILED_DATA_L, no_data_list)
        # return
        return no_data_list, key_list, ecg_inst

    def write_numpy(self, ecgdrawing: ECGDrawing,
                    target_path: str = '.', write_failed: bool = True,
                    kwargs_reader: Dict[Any, Any] | None = None,
                    kwargs_drawing: Dict[Any, Any] | None = None,
                    kwargs_savefig: Dict[Any, Any] | None = None,
                    ) -> Self:
        '''
        Extracts xml files, and write these one by one to numpy files
        using a supplied `ECGDrawing` instance.

        Parameters
        ----------
        ecgdrawing : ECGDrawing
            An instance of the ECGDrawing data class.
        target_path : `str`, default '.'
            The full path where the npzs should be written to.
        write_failed : `bool`, default `True`
            Whether to write a text file to disk containing the failed file
            names.
        kwargs_*: dict [`any`, `any`], default `NoneType`
            dictionaries with keyword arguments for the `plt.savefig`,
            ECGDrawing, or ECGXMLReader instances.

        Attributes
        ----------
        target_path : `str`
            The directory or tar file path were the files are written to.
        NoDataList : `list` [`str`]
            A list of xml files without a waveform_array.

        Returns
        -------
        self : `ECGXMLTable` instance
            Returns the class instance with updated attributes.

        Notes
        -----
        The xml UID instance will be used as file name for the npzs.

        Raises
        ------
        NotADirectoryError or PermissionError
            If the target directory does not exist or is not writable.
        '''
        # #### check input and set constants
        is_type(kwargs_reader, (type(None), dict))
        is_type(kwargs_drawing, (type(None), dict))
        is_type(kwargs_savefig, (type(None), dict))
        is_type(ecgdrawing, ECGDrawing)
        is_type(target_path, (pathlib.PosixPath, str))
        is_type(write_failed, bool)
        # check readability
        _check_presence(target_path)
        _check_readable(target_path)
        # map None to dict
        kwargs_savefig, kwargs_reader, kwargs_drawing = assign_empty_default(
                [kwargs_savefig, kwargs_reader, kwargs_drawing], dict)
        # map info type to type of plot
        if getattr(self, PDNames.INFO_TYPE) == PDNames.INFO_TYPE_MET:
            raise AttributeError('`info` should be {0} or {1}.'.format(
                PDNames.INFO_TYPE_MED, PDNames.INFO_TYPE_RTM))
        elif getattr(self, PDNames.INFO_TYPE) == PDNames.INFO_TYPE_ALL:
            signal_type = PDNames.WAVETYPE_RHYTHM
        else:
            signal_type = getattr(self, PDNames.INFO_TYPE)
        # #### create target path
        if target_path == '.':
            target_path = os.getcwd()
        # assign to self
        target = target_path
        print(target)
        setattr(self, PDNames.WRITE_ECG_PATH, target)
        # #### check if __call__ has been run
        if not hasattr(self, PDNames.CPATH_L):
            raise NotCalledError()
        # #### extract xml data
        key_list, no_data_list = [[] for _ in range(2)]
        # loop over individual xml files
        for p in tqdm(getattr(self, PDNames.CPATH_L), desc="Processing", unit="file"):
            no_data_list, key_list, ecg_inst = self._write_internal(
                path=p, no_data_list=no_data_list, key_list=key_list,
                **kwargs_reader,
                )
            if isinstance(ecg_inst, str):
                continue
            artist = ecgdrawing(ecgreader=ecg_inst, wave_type=signal_type,
                                **kwargs_drawing)

            # Check if median waveform is present, if not, skip NPZ creation
            if getattr(ecg_inst, PDNames.MEDIAN_PRESENT) is False:
                print(f"The file {re.sub(r"[ ,\-\(\)\{\}]", '_', os.path.basename(key_list[-1]))} does not contain a median waveform. ECG recording probably failed. Skipping numpy creation.")
                continue

            # Check if motion artefacts are present, if so, skip NPZ creation.
            if not artist.check_all_ecg_leads_threshold():
                print(f"One or more ECG leads exceed the threshold. Skipping numpy creation for {re.sub(r"[ ,\-\(\)\{\}]", '_', os.path.basename(key_list[-1]))}.")
                continue

            # Convert matplotlib plot to NumPy
            arr = artist.to_numpy(crop=False)

            # Save as npz
            filename_npy = re.sub(
                r"[ ,\-\(\)\{\}]", '_', os.path.basename(key_list[-1])) + '.npz'
            np.savez_compressed((os.path.join(target, filename_npy)), image=arr)

            plt.close(artist.fig)

        # #### write failed files, note not compressing these
        DELIM = '\t'
        if write_failed is True:
            # adding the reason for failing
            total_failures = [
                (p, PDNames.SKIP_PERMISSIONS) for p in
                getattr(self, PDNames.FPATH_L)] + [
                (p, PDNames.SKIP_DATA) for p in
                    getattr(self, PDNames.FAILED_DATA_L)]
            # writing to text file
            with open(os.path.join(target, PDNames.FAILED_FILE), 'w') as file:
                for p, cause in total_failures:
                    file.write(p + DELIM + cause + "\n")
        # #### return
        return self
