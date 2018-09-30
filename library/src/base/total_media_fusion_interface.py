import logging
from time import time

from .cwbook import ConfigWB
from .preprocess import Preprocess


class TotalMediaFusionInterface(object):

    def __init__(self, cwbook_obj, preprocess_obj, config_flnm):
        self._logger = logging.getLogger(__name__)

        assert isinstance(cwbook_obj, ConfigWB)

        self._cwbook = cwbook_obj

        assert isinstance(preprocess_obj, Preprocess)

        self._pre_obj = preprocess_obj

        assert isinstance(config_flnm, str)
        self.flnm = config_flnm

    @property
    def preprocess(self):
        return self._pre_obj

    @preprocess.setter
    def preprocess(self, obj):

        assert isinstance(obj, Preprocess)

        self._pre_obj = obj

    @property
    def cwbook(self):
        return self._cwbook

    @cwbook.setter
    def cwbook(self, obj):

        assert isinstance(obj, ConfigWB)

        self._cwbook = obj

    def configure(self):
        """
        :return: self
        """
        logger = self._logger

        preprocess = self.preprocess

        cwbook = self.cwbook

        logger.info('Setting up configurations')

        try:

            cwbook.iw_rec_vars = preprocess.iw_rec_vars
            cwbook.iw_don_vars = preprocess.iw_don_vars
            cwbook.se_eval_vars_initial = preprocess.linking_variables_se_init
            cwbook.se_eval_vars_final = preprocess.linking_variables_se_final
            cwbook.lv = preprocess.linking_variables_master
            cwbook.lv_iw = preprocess.linking_variables_iw
            cwbook.lv_pc = preprocess.linking_variables_pc

            logger.info('Successfully set configurations')

        except Exception as e:
            logger.fatal(e, exc_info=True)

        self.cwbook = cwbook

        return self

    def save_config(self):
        """
        :param flnm:
        :return:
        """
        cwbook = self.cwbook

        return self

    def preprocess_run(self, params_donors_recips, params_soft_cal, params_update_donors_recips, calibrate_donor):
        """
        :param params_donors_recips:
        :param params_soft_cal:
        :param params_update_donors_recips:
        :param calibrate_donor:
        :return: self
        """
        logger = self._logger

        preprocess = self.preprocess

        logger.info('Creating Donor and Recipient Datasets')

        start = time()

        try:
            # creates the cph_donors and cph_recips
            if ('donors' in params_donors_recips.keys()) and ('recips' in params_donors_recips.keys()):
                preprocess.donors = params_donors_recips.get('donors')
                preprocess.recips = params_donors_recips.get('recips')
            else:

                preprocess.create_donors_recips(**params_donors_recips)

                logger.info('Successfully created the Donor and Recipient Datasets')

            logger.info('It took: {} seconds'.format(time() - start))

            logger.info('Donor sample size is {}'.format(preprocess.donors.shape[0]))
            logger.info('Recipient sample size is {}'.format(preprocess.recips.shape[0]))

            self.preprocess = preprocess

        except AssertionError as e:
            logger.fatal(e, exc_info=True)
            return False

        logger.info('Saving the linking variables, sample evaluation variables, \
        and importance weights variables to configuration parameters')

        # we update the configuration workbook with different parameters for our fusion

        logger.info('Creating Soft Calibration Targets')

        start = time()

        try:

            # create the soft calibration targets and impressions
            preprocess.create_soft_cal_targets(**params_soft_cal)

            logger.info('Successfully created the Soft Calibration Targets')
            logger.info('It took: {} seconds'.format(time() - start))

            logger.info('The number of targets is {}'.format(preprocess.soft_cal_targets['var'].nunique()))

            self.preprocess = preprocess

        except AssertionError as e:
            logger.fatal(e, exc_info=True)
            return False

        if calibrate_donor:
            preprocess.calibrate_donor_weights()
        else:
            preprocess.calibrate_recip_weights()

        logger.info('Updating donors and recipients based on previous linkages')

        start = time()

        try:

            preprocess.update_donor_recips(**params_update_donors_recips)

            logger.info('Successfully determined old linkages')
            logger.info('It took: {} seconds'.format(time() - start))

        except AssertionError as e:

            self.preprocess = preprocess

            logger.fatal(e, exc_info=True)
            return False

        self.preprocess = preprocess

        try:
            self.configure()
        except Exception as e:
            logger.fatal(e, exc_info=True)
            logger.fatal('One of the linking variable parameters were not set during this step.')
            return False

        self.save_config()

        return True

    def fusion_run(self):

        return True

    def run(self, params_donors_recips, params_soft_cal, params_update_donors_recips, calibrate_donor):
        """
        :param params_donors_recips:
        :param params_soft_cal:
        :param params_update_donors_recips:
        :param calibrate_donor:
        :return:
        """
        success = self.preprocess_run(params_donors_recips, params_soft_cal, params_update_donors_recips,
                                      calibrate_donor)

        logger = self._logger

        if success:
            try:
                logger.info('Running the PyFusion App')
                self.fusion_run()
                return True
            except Exception as e:
                logger.fatal(e, exc_info=True)
                return False

        else:
            return False
