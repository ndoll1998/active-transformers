import re
import logging
import datasets
import warnings
from collections import defaultdict
from typing import List, Dict, Union, Any

logger = logging.getLogger(__name__)

class ConllFormatBuilder(datasets.GeneratorBasedBuilder):

    # specify the line format, i.e. map each entry of a line
    # to a dataset feature
    FORMAT:List[Union[str, None]] = None
    # specify feature types for each feature listed in the
    # format
    FEATURE_TYPES:Dict[str, Any] =None

    @property
    def _features(self) -> datasets.Features:
        # build dataset features
        return datasets.Features(
            {
                feature_name: datasets.Sequence(feature_type)
                for feature_name, feature_type in type(self).FEATURE_TYPES.items()
            }
        )

    def _generate_examples(self, fpaths):
        for fpath in fpaths:
            logger.info("Generating Examples from %s" % fpath)
            # shorthands for class members
            F = type(self).FORMAT

            with open(fpath, 'r', encoding='latin-1') as f:

                guid = 0
                item = defaultdict(list)

                for line in map(str.strip, f):
                    # replace all spaces with simple whitespaces
                    # this includes tabs and multiple whitespaces
                    # needed for reliable splitting later
                    line = re.sub(r"\s+", " ", line)

                    if line.startswith('-DOCSTART-') or len(line) == 0:
                        # check if item is non-empty
                        if len(item) > 0 and len(next(iter(item.values()))) > 0:
                            # yield item
                            yield guid, item
                        
                        # reset item
                        guid += 1
                        item = defaultdict(list)

                    else:
                        # split line
                        features = line.strip().rsplit(' ', len(F)-1)
                        # check if line is valid
                        if len(features) != len(F):
                            warnings.warn("Found mismatch between file line and specified format!", UserWarning)
                            continue
                        # update current item
                        for field, value in zip(F, line.strip().rsplit(' ', len(F)-1)):
                            # check if field is not ignored
                            if (field is None):
                                continue
                            item[field].append(value.strip())
                 
                # yield last item
                if len(item) > 0 and len(next(iter(item.values()))) > 0:
                    yield guid, item

