# This program, TimeLeftBeforeSunset, creates abacuses to estimate the time
# remaining before sunset, using only your fingers.
#
# Copyright (C) December 2023
# Author : Xoutron
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from pvlib import solarposition
import pandas as pd
import numpy as np
import math
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from uncertainties import ufloat
from uncertainties import unumpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

lang = 'fr' # 'fr' (French) or 'en' (English)

#---
#  Translation function
#---
def translate_city(city, lang):
    if lang == 'en':
        if city == 'Trondheim (Norvège)':
            city = 'Trondheim (Norway)'
        elif city == 'Édimbourg (Écosse)':
            city = 'Edinburgh (Scotland)'
        elif city == 'Caen':
            city = 'Caen (France)'
        elif city == 'Grenoble':
            city = 'Grenoble (France)'
        elif city == 'Syracuse (Sicile)':
            city = 'Syracuse (Sicily)'
        elif city == 'Quito (Équateur)':
            city = 'Quito (Ecuador)'
    # elif lang == 'fr': will stay as it was
    return city

#---
#  Computing functions
#---
def get_time_pd(time, month):
    hour = math.floor(time)
    minute = math.floor((time - math.floor(time))*60)
    second = (time - hour - minute/60)*3600
    time_pd = pd.DatetimeIndex(data = [str(month) + "/21/2023 "
                                       + str(hour) + ":"
                                       + str(minute) + ":"
                                       + str(second)], tz = 'etc/UTC')
    return time_pd
def get_solar_angular_radius(month):
    time_pd = get_time_pd(12, month) # Retrieve at noon (approximate)
    # Earth-sun distance in A.U. (astronominal unit), through
    # https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.nrel_earthsun_distance.html
    earth_sun_distance_AU = solarposition.nrel_earthsun_distance(time_pd)[0]
    # 1 A.U. = 149597870.7 km [https://en.wikipedia.org/wiki/Astronomical_unit]
    earth_sun_distance = earth_sun_distance_AU * 149597870.7
    # Solar radius extracted from: Emilio, Marcelo; Kuhn, Jeff R.; Bush,
    # Rock I.; Scholl, Isabelle F. (2012), "Measuring the Solar Radius from
    # Space during the 2003 and 2006 Mercury Transits", The Astrophysical
    # Journal, 750 (2): 135, https://arxiv.org/pdf/1203.4898.pdf
    solar_radius = 696342 # [km]
    # https://en.m.wikipedia.org/wiki/Angular_diameter#Formula
    angle = np.arcsin(solar_radius/earth_sun_distance)
    # Convert that angle from radians to degree
    return angle*180/math.pi
def get_zenith(time, month, lat):
    time_pd = get_time_pd(time, month)
    lon = 0 # The sunset's speed does not depend on longitude. Greenwich's fine!
    # https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html
    solar_zenith = solarposition.get_solarposition(time_pd, lat, lon)
    # 'apparent_zenith' takes atmospheric refraction into account, with
    # average atmospheric conditions
    return solar_zenith['apparent_zenith']

def wrapper_get_zenith(time, month, lat):
    return get_zenith(time, month, lat)[0]

def diff_to_angle(time, angle, month, lat):
    return get_zenith(time, month, lat) - angle

#---
#  Compute angles for each finger, from five measurements [cm]
#---
# I don't know the width of your fingers, the length of your arms, etc., so
# I've used mine
hand_measurements = np.array([[2.733, 3,    3,     2.95,  3.033], # above index
                              [4.9,   5,    5.1,   5.1,   5.066], # index
                              [7.2,   7.4,  7.433, 7.4,   7.4], # middle finger
                              [9.35,  9.67, 9.633, 9.7,   9.7], # annulary
                              [11.05, 11.5, 11.5,  11.55, 11.53]]) # little
# Align using the reference point (above index)
finger_widths = hand_measurements[1:] - hand_measurements[:-1]
# Average over the five measurements
finger_widths_mean = np.mean(finger_widths, axis = 1)
# Compute standard deviation with those five measurements
finger_widths_std = np.std(finger_widths, axis = 1)
# Use 2 standard deviations
finger_widths = unumpy.uarray(finger_widths_mean, 2*finger_widths_std)
# Schematic side view of situation :
#                                        __
#                ---------| index          \
#       ---------         | middle finger  | opposite
#    ---angle             | annulary       | sides
#  *)---------------------| little finger__/
# eye   adjacent side   hand
#
# Using the small angle (Gauss) approximation, tan(angle) =~ angle [radians]
# and moreover, tan = opposite/adjacent. Also, converting radians into degrees.
Eye_to_hand = ufloat(55.5, 0.5) # cm (adjacent)
finger_angles = finger_widths/Eye_to_hand*180/math.pi
finger_angles = np.flip(finger_angles) # little finger first
angles = []
nhand = 3 # number of hands, obtained by stacking them alternately
for hand in range(nhand):
    angles = np.append(angles,
                       # Add a quarter degree uncertainty per additional hand
                       finger_angles + np.array([1]*4)*ufloat(0, hand*0.25))
# Cumulative angle, from little finger alone (at horizon level) to all fingers
finger_angles = np.cumsum(angles)
#---
#  Computational parameters
#---
cities = [(63.429722, 'Trondheim (Norvège)'), # https://en.wikipedia.org/wiki/Trondheim
          (55.953333, 'Édimbourg (Écosse)'), # https://en.wikipedia.org/wiki/Edinburgh
          (49.1826, 'Caen'), # https://fr.wikipedia.org/wiki/Caen
          (45.1859, 'Grenoble'), # https://fr.wikipedia.org/wiki/Grenoble
          (37.069167, 'Syracuse (Sicile)'), # https://en.wikipedia.org/wiki/Syracuse,_Sicily
          (-0.22, 'Quito (Équateur)')] # https://en.wikipedia.org/wiki/Quito
# Compute from June to December : January is equivalent to November,
# February to October, March to September, etc.
minmonth = 6
maxmonth = 12
#---
#  Compute durations before sunset
#---
durations = {}
# Latitude in degree
for lat, city in cities:
    durations[city] = []
    print(city)
    for month in range(minmonth, maxmonth+1):
        print("month=", month)
        # Following https://aa.usno.navy.mil/faq/RST_defs, sunset is defined to
        # occur when the top of the solar disk appears to be tangent to the
        # horizon, for an observer at sea level with a level, unobstructed
        # horizon. That is, the center of the Sun is below the horizonal plane,
        # by an extent equal to the angular radius of the Sun.
        sunset_angle = 90 + get_solar_angular_radius(month)
        # Convert elevations to zenith angles for which we want to compute time
        angles = np.concatenate(([sunset_angle], 90 - finger_angles))
        # Searching solar noon, which is the minimum zenith angle reached
        # during the day
        solar_noon = minimize_scalar(wrapper_get_zenith, args=(month, lat),
                                     bounds=(10, 17), # hours search limits
                                     method='Bounded',
                                     options = {'xatol': 1/3600}) # within 1sec
        solar_noon_time = solar_noon.x
        solar_noon_angle = solar_noon.fun
        times = []
        for angle in angles:
            for err in [+1, -1]: # Compute for both maximal and minimal value
                angle_extremum = (unumpy.nominal_values(angle)
                                  + err*unumpy.std_devs(angle))
                if angle_extremum < solar_noon_angle:
                    # That angle isn't reached at all on that day, even at noon
                    time = float('nan')
                else:
                    time = brentq(diff_to_angle, solar_noon_time, 23,
                                   args = (angle_extremum, month, lat),
                                   maxiter = 100, xtol = 1/3600) # within 1 sec
                times.append(time)
        # Rearrange, to put them in pairs (minimum/maximum)
        times = np.reshape(times, (-1, 2))
        # Compute durations (in minutes) before sunset (the latter being first
        # line)
        duration = (times[0] - times)[1:]*60
        print(np.flip(duration, axis=0)) # flip to get little finger at the bottom
        durations[city].append(duration)
    durations[city] = np.array(durations[city])
#---
#  Plotting
#---
if not os.path.exists('./Plots'):
    os.mkdir('./Plots')
colors = {}
colors['Trondheim (Norvège)'] = 'tab:blue'
colors['Édimbourg (Écosse)'] = 'tab:orange'
colors['Caen'] = 'tab:green'
colors['Grenoble'] = 'tab:red'
colors['Syracuse (Sicile)'] = 'tab:pink'
colors['Quito (Équateur)'] = 'tab:cyan'
fingers = {}
prefingers = {}
handstr = {}
if lang == 'fr':
    fingers[0] = 'Auriculaire'
    fingers[1] = 'Annulaire'
    fingers[2] = 'Majeur'
    fingers[3] = 'Index'
    prefingers[0] = "de l'auriculaire"
    prefingers[1] = "de l'annulaire"
    prefingers[2] = "du majeur"
    prefingers[3] = "de l'index"
    handstr[0] = "première main"
    handstr[1] = "seconde main"
    handstr[2] = "troisième main"
elif lang == 'en':
    fingers[0] = 'little finger'
    fingers[1] = 'annulary'
    fingers[2] = 'middle finger'
    fingers[3] = 'index'
    handstr[0] = "first hand"
    handstr[1] = "second hand"
    handstr[2] = "third hand"
for hand in range(nhand):
    for ifinger in range(4):
        nfinger = (ifinger + 1) + hand*4
        labels = []
        fig, ax = plt.subplots()
        for lat, city in cities:
            # Remove some cities for overlapping higher elevations
            if not ((city == 'Caen' and hand > 0) or
                    (city == 'Syracuse (Sicile)' and hand > 1)):
                ax.fill_between(np.arange(minmonth, maxmonth+1),
                                durations[city][:, nfinger-1, 0],
                                durations[city][:, nfinger-1, 1],
                                color = colors[city])
                label = translate_city(city, lang)
                label = matplotlib.lines.Line2D([], [], color = colors[city],
                                                label = label)
                labels.append(label)
        ax.grid()
        # Add a constant dashed line on 15-30-45...minutes (the usual estimate)
        usual_number = nfinger*15
        ax.plot([minmonth, maxmonth], [usual_number, usual_number],
                color = 'black', linestyle = (0, (5, 10))) # loosely dashed
        if lang == 'fr':
            title = ("Soleil arrivant au haut " + prefingers[ifinger] + ", "
                     + handstr[hand] + ' (' + str(nfinger) + ' doigt')
        elif lang == 'en':
            title = ('Sun reaching top of ' + fingers[ifinger] + ', '
                     + handstr[hand] + ' (' + str(nfinger) + ' finger')
        if nfinger > 1:
            title = title + 's'
        ax.set_title(title + ')')
        if lang == 'fr':
            month_labels = ['Juin',
                            'Juillet\nMai',
                            'Août\nAvril',
                            'Sept\nMars',
                            'Oct\nFév',
                            'Nov\nJanv',
                            'Déc']
        elif lang == 'en':
            month_labels = ['June',
                            'July\nMay',
                            'August\nApril',
                            'Sept\nMarch',
                            'Oct\nFeb',
                            'Nov\nJan',
                            'Dec']
        ax.set_xticks(np.arange(minmonth, maxmonth + 1))
        ax.set_xticklabels(month_labels)
        ax.set_xlim([minmonth, maxmonth])
        # https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-the-x-or-y-axis
        if nfinger == 1 or nfinger == 2 or nfinger == 3 or nfinger == 4:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) # minutes
        elif nfinger == 5:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(10)) # minutes
        elif nfinger == 6 or nfinger == 7 or nfinger == 8:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(20)) # minutes
        if lang == 'fr':
            ax.set_xlabel('21ème jour du mois de...')
            ax.set_ylabel('Minutes avant le coucher du soleil')
        elif lang == 'en':
            ax.set_xlabel('21st day of the month of...')
            ax.set_ylabel('Minutes before sunset')
        if nfinger < 10:
            ax.legend(handles = labels, loc = 'upper center', framealpha = 1)
        else:
            ax.legend(handles = labels, loc = 'upper right', framealpha = 1)
        plt.tight_layout()
        if lang == 'fr':
            txt = ('Temps restant avant le coucher du soleil © 2023 par '
                   + 'Xoutron est sous licence CC BY-NC-SA 4.0.\n'
                   + 'Pour obtenir une copie de cette licence, voir '
                   + 'http://creativecommons.org/licenses/by-nc-sa/4.0/')
        elif lang == 'en':
            txt = ('Time Left Before Sunset © 2023 by Xoutron is licensed '
                   + 'under CC BY-NC-SA 4.0.\nTo view a copy of this license, '
                   + 'visit http://creativecommons.org/licenses/by-nc-sa/4.0/')
        plt.text(1.01, 0.5, txt, fontsize = 5, rotation = 'vertical',
                 horizontalalignment = 'left', verticalalignment = 'center',
                 transform = ax.transAxes)
        filename = lang + '-' + str(nfinger) + '-' + fingers[ifinger]
        fig.savefig('Plots/' + filename + '.pdf')
        fig.savefig('Plots/' + filename + '.png', dpi = 400)
        del fig, ax
#---
#  Printing the tables, rounded to the closest minute
#---
if not os.path.exists('./Output'):
    os.mkdir('./Output')
rounded_durations = {}
for lat, city in cities:
    print(city)
    rounded_durations[city] = []
    nmonth = np.shape(durations[city])[0]
    nangle = np.shape(durations[city])[1]
    for month in range(nmonth):
        for angle in range(nangle):
            minimum = durations[city][month, angle, 0]
            maximum = durations[city][month, angle, 1]
            if math.isnan(minimum) or math.isnan(maximum):
                string = 'NaN'
            else:
                string = str(round(minimum)) + '~' + str(round(maximum))
            rounded_durations[city].append(string)
    rounded_durations[city] = np.reshape(rounded_durations[city],
                                         (nmonth, nangle))
    # Turn so that :
    # * little finger is at the bottom (just like in real life),
    # * going from left to right means going from September to December.
    rounded_durations[city] = np.flip(np.transpose(rounded_durations[city]),
                                      axis=0)
    #---
    #  A version complete but bulky, suited only for complete references
    #---
    index = np.arange(np.shape(rounded_durations[city])[0], 0, -1)
    columns = ['Juin', 'Juil<br>Mai', 'Août<br>Avr', 'Sept<br>Mars',
               'Oct<br>Fév', 'Nov<br>Jan', 'Déc']
    df = pd.DataFrame(rounded_durations[city], index = index,
                      columns = columns)
    # Prepare filename to which data will be exported
    filename = translate_city(city, 'en').replace(' ', '_')
    filename = filename.replace('(', '').replace(')', '')
    filename = 'Output/' + filename + '.md'
    # Export data to text files
    this_file = open(filename, 'w')
    this_file.write(df.to_markdown().replace('-|', ':|')) # centering columns
    this_file.close()
    #---
    #  A more streamlined version
    #---
    columns = ['Juin', 'Juil\nMai', 'Août\nAvr', 'Sept\nMars', 'Oct\nFév',
               'Nov\nJan', 'Déc']
    df = pd.DataFrame(rounded_durations[city], index = index,
                      columns = columns)
    print(df[['Juin', 'Sept\nMars', 'Déc']].to_markdown())
