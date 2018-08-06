#!/usr/bin/env python
# coding: utf-8
# Date  : 2018-08-06 11:40:23
# Author: b4zinga
# Email : b4zinga@outlook.com
# Func  : operate files

import re
from collections import Counter


class FA:
    """FA file is the protein FASTA file download for NCBI.
    BCBI(https://www.ncbi.nlm.nih.gov/genomes/FLU/Database/nph-select.cgi#mainform).
    """
    def __init__(self, fa):
        """Initialization.
        time_format_err is the record of false time formate in FA file.

        """
        self.fa = fa
        self.time_format_err = []

    def getPlaceAndTime(self):
        """

        return a list like that: [('South Carolina', '2012/10/04'),]
        """
        place_time = []

        # sample:
        # >AEK85498 A/Taiwan/90252/2011 2011/02/ HA
        # >BAI48182 A/Nagasaki/HA-29/2009 2009/07/30 HA
        # >ABK40557 A/Memphis/27/1983 1983// HA
        regex = ">.*?\s\w/(.*?)/.*?/\d+\s(\d+/\d+/\d+)\s.*?$"  # low
        # regex = '>.*?\s\w/(.*?)/.*?/\d+\s+(\d+/\d*/\d*)\s+HA'  # middle
        # regex = '>.*?\s\w/(.*?)/\d+\s+(\d+/\d*/\d*)\s.*?$'     # high

        with open(self.fa) as file:
            for f in file:
                if f.startswith('>'):
                    items = re.findall(regex, f)
                    if items:
                        place_time.append(items[0])
                    else:
                        self.time_format_err.append(f)

        return place_time

    def getDiffPlaceTime(self):
        """get the amount of flu erupting in the same place and time.
        return a dict like that: {('New Jersey', '1976//'): 3, ('HaNoi', '2003/01/26'): 1,}
        """
        return dict(Counter(self.getPlaceAndTime()))

    def getTimeFormatErr(slef):
        return self.time_format_err

    def getVirusAmino(self):
        """
        """
        pass


class City:
    """City file involves the list of global cities relating to longitude and latitude.
    """
    def __init__(self, city):
        """Initialization.
        city_format_err is the record of false format in city file.
        """
        self.city = city
        self.city_format_err = []

    def getCityLocation(self):
        """get all the city names and corresponding longitude and latitude in city file.
        return a dict like that: {'Tientsin': ('39.084158', '117.200983'),}
        """
        city_location = {}
        # sample:
        #  58.001985     56.257287      Perm
        #  53.6126505    12.4295953     Germany-MV(Mecklenburg-Vorpommern)
        regex = ".?([-]?\d+[\.]?\d+)\s+([-]?\d+[\.]?\d+)\s+(.*?)[(\n]"
        with open(self.city) as file:
            for f in file:
                items = re.findall(regex, f)
                if items:
                    for item in items:
                        city_location[item[2]] = (item[:2])
                else:
                    self.city_format_err.append(f)

        return city_location

    def getCityFormatErr(self):
        """return a list of false format of longitude and latitude.
        """
        return self.city_format_err

    def addCityIntoFile(self, jing, wei, place):
        """add longitude and latitude to city file.
        """
        info = ' ' + wei + '    ' +jing + '    ' + place + '\n'
        with open(self.city, 'a') as cf:
            cf.write(info)
        cf.close()


class Operation(FA, City):
    """Some methods based on the FA file and City file."""
    def __init__(self, fa, city):
        """Initialization.
        match_failed_record is records that city names can be found in FA file, 
        but corresponding longitude and latitude can not be found in city file.
        """
        super().__init__(fa)
        super(FA, self).__init__(city)

        self.match_failed_record = []

    def getExplosionDetail(self):
        """return a list like that: [['Singapore', '2009/09/23', ('1.352083', '103.819836'), 3],]
        """
        details = []
        place_and_time = self.getDiffPlaceTime()
        place_jing_wei = self.getCityLocation()
        for pt, num in place_and_time.items():
            try:
                details.append([pt[0], pt[1], place_jing_wei[pt[0]], num])
            except KeyError:
                self.match_failed_record.append((pt,num),)

        return sorted(details, key=lambda item: item[1])

    def getMatchFailedRecord(self):
        """match_failed_record is records that city names can be found in FA file, 
        but corresponding longitude and latitude can not be found in city file.
        """
        return self.match_failed_record

    def judgeIntegrity(self):
        """judging the intergrity of city file, which means whether city names in
        FA file can be matched with corresponding longitude and latitude in city
        file or not.
        If matched, return False, else return unmatched city names.
        """
        place_and_time = self.getPlaceAndTime()
        city_location = self.getCityLocation()
        no_exist_place = set()
        for pt in place_and_time:
            if pt[0] not in city_location.keys():
                no_exist_place.add(pt[0])
        if len(no_exist_place):
            return no_exist_place
        else:
            return False

    def getDetailByTime(self):
        """get corresponding information according to the order of flu erruption.
        return a dict like that : 
        {'2010/09/16': [['Sydney', '2010/09/16', ('-33.8674869', '151.2069902'), 2], ['Bangkok', '2010/09/16', ('13.7522222', '100.4938889'), 1]], }
        and the start time and end time of influenza.
        """
        details_by_time = {}
        virus = self.getExplosionDetail()
        start_time, end_time = virus[0][1], virus[-1][1]
        for vir in virus:
            details_by_time[vir[1]] = []
        for vir in virus:
            details_by_time[vir[1]].append(vir)

        return details_by_time, start_time, end_time



if __name__ == '__main__':
    op = Operation('./data/H1N1_1965-2012.fa', './data/LatLon_H1N1.txt')
    print(op.getDetailByTime())
