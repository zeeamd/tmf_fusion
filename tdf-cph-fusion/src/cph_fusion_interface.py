import logging

from library.src.base.total_media_fusion_interface import TotalMediaFusionInterface

from .cph_cwbook import CPHWBook
from .cph_preprocess import CPHPreprocess


class CPHFusion(TotalMediaFusionInterface):

    def __init__(self):
        TotalMediaFusionInterface.__init__(self, CPHWBook(), CPHPreprocess(), 'cph_config_file.json')
        self._logger = logging.getLogger(__name__)

    def save_config(self):

        logger = self._logger

        logger.info('Saving configurations to file')

        TotalMediaFusionInterface.__init__.save_config(self)

        return self

    def run(self, params_donors_recips, params_soft_cal, params_update_donors_recips):
        """
        :param params_donors_recips:
        :param params_soft_cal:
        :param params_update_donors_recips:
        :return: self
        """

        logger = self._logger

        logger.info('Starting CPH Fusion')

        success = TotalMediaFusionInterface.run(self, params_donors_recips, params_soft_cal,
                                                params_update_donors_recips, True)

        if success:
            logger.info('Successfully ran through all the steps')

        else:
            logger.fatal('This iteration failed.')

        return success
