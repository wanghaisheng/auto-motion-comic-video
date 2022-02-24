import os
from posixpath import sep
import re
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome import options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import nltk
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, WebDriverException

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
import glob
import platform
import random
from .language_detect import prediction
# Make the Pool of workers
regexpatternforurl=r"((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)"


def getwebdriver_chrome():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation"])
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_experimental_option('useAutomationExtension', False)

    prefs = {'profile.default_content_setting_values.automatic_downloads': 1}
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument('lang=zh-CN,zh,zh-TW,en-US,en')
    chrome_options.add_argument(
        'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36')
    chrome_options.add_argument("proxy-server=socks5://127.0.0.1:1080")
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-logging"])

    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_experimental_option("prefs", {
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    if 'Windows' in platform.system():
        web_driver = webdriver.Chrome(executable_path="assets/driver/chromedriver.exe", chrome_options=chrome_options)
        # print('windows system chrome driver ',web_driver)

    else:
        web_driver = webdriver.Chrome(executable_path="assets/driver/chromedriver", chrome_options=chrome_options)
    return web_driver


def tiny_file_rename(newname, folder_of_download, time_to_wait=60):
    finished = is_download_finished(folder_of_download)

    while not finished:

        finished = is_download_finished(folder_of_download)
        # print(' download finished?',finished)
        time.sleep(0.2)

        if finished == True:
            break

    if os.path.exists(folder_of_download+os.sep+'narration.mp3'):
        filename = max([f for f in os.listdir(folder_of_download)],
                       key=lambda xa:   os.path.getctime(os.path.join(folder_of_download, xa)))
        print("file is there",filename,'rename ',os.path.join(folder_of_download, filename),'to',os.path.join(folder_of_download, newname))

        if os.path.exists(os.path.join(folder_of_download, newname)):
            os.remove(os.path.join(folder_of_download, newname))
            os.rename(os.path.join(folder_of_download, filename),
                    os.path.join(folder_of_download, newname))            
        else:
            os.rename(os.path.join(folder_of_download, filename),
                    os.path.join(folder_of_download, newname))
        print('narration.mp3 renamed to',os.path.join(folder_of_download, newname))
    return os.path.join(folder_of_download, newname)

def is_download_finished(temp_folder):
    firefox_temp_file = sorted(Path(temp_folder).glob('*.part'))
    chrome_temp_file = sorted(Path(temp_folder).glob('*.crdownload'))
    downloaded_files = sorted(Path(temp_folder).glob('*.*'))

    if (len(firefox_temp_file) == 0) and \
       (len(chrome_temp_file) == 0) and \
       (len(downloaded_files) >= 1):
        return True
    else:
        return False
def TextCorrectorbeforeTTS(txt):
	result = re.sub(regexpatternforurl, " ", txt)

	result = result.replace("{w}","...")
	result = result.replace("{i}","")
	result = re.sub(r"\s*{.*}\s*", " ", result)
	result = re.sub(r"\s*\[.*\]\s*", " ", result)
    
	# print(result)
	return result

proxies = {
    'http': 'socks5://127.0.0.1:1080',
    'https': 'socks5://127.0.0.1:1080'
}

supported_languages = { # as defined here: http://msdn.microsoft.com/en-us/library/hh456380.aspx
  'da' : 'Danish',
  'nl' : 'Dutch',
  'en' : 'English',
  'fr' : 'French',
  'de' : 'German',
  'it' : 'Italian',
  'is' : 'Iceland',
  'no' : 'Norwegian',
  'pt' : 'Portuguese',
  'ru' : 'Russian',
  'es' : 'Spanish',
  'sv' : 'Swedish',
  'tr' : 'Turkish',
  'ro' : 'Romanian',
  'ja' : 'Japanese',
  'pl' : 'Polish',
}

#polly voice name map 
male_languages = { 
  'da' : 'Mads',
  'nl' : 'Ruben',
  'en' : 'Joey',
  'fr' : 'Mathieu',
  'de' : 'Hans',
  'it' : 'Giorgio',
  'is' : 'Karl',
  'no' : 'Liv',
  'pt' : 'Cristiano',
  'ru' : 'Maxim',
  'es' : 'Enrique',
  'sv' : 'Astrid',
  'tr' : 'Filiz',
  'ro' : 'Carmen',
  'ja' : 'Mizuki',
  'pl' : 'Jacek',
}

female_languages = { 
  'da' : 'Naja',
  'nl' : 'Lotte',
  'en' : 'Joanna',
  'fr' : 'Céline',
  'de' : 'Marlene',
  'it' : 'Carla',
  'is' : 'Dóra',
  'no' : 'Liv',
  'pt' : 'Inês',
  'ru' : 'Tatyana',
  'es' : 'Conchita',
  'sv' : 'Astrid',
  'tr' : 'Filiz',
  'ro' : 'Carmen',
  'ja' : 'Mizuki',
  'pl' : 'Ewa',
}

def ttsmp3_polly_tts(text, character_name,ttsdir,index):
    # print('tts',text)
    

# Add a break
# Mary had a little lamb <break time="1s"/> Whose fleece was white as snow.
# Emphasizing words
# I already told you I <emphasis level="strong">really like </emphasis> that person. 
    TTSMP3_URL = "https://ttsmp3.com/makemp3_new.php"

    print('======which language text is to tts=====\n',text)
    language = prediction(text)
    if language=='en':
        pass
    elif language=='fr':
        pass
    character_idlist=[]
    AllowedVoiceList_zh = [
                        "Zeina", "Zhiyu", "Naja", "Mads", "Lotte", "Ruben",
                        "Nicole", "Russell", "Amy", "Emma", "Brian", "Aditi", "Raveena",
                        "Ivy", "Joanna", "Kendra", "Kimberly", "Salli", "Joey", "Justin",
                        "Matthew", "Geraint", "Celine", "Mathieu", "Chantal", "Marlene",
                        "Vicki", "Hans", "Dora", "Karl", "Carla", "Bianca", "Giorgio",
                        "Mizuki", "Takumi", "Seoyeon", "Liv", "Ewa", "Maja", "Jacek", "Jan",
                        "Vitoria", "Ricardo", "Ines", "Cristiano", "Carmen", "Tatyana", "Maxim",
                        "Conchita", "Lucia", "Enrique", "Mia", "Penelope", "Miguel", "Astrid", "Filiz",
                        "Gwyneth"
                        ]   

    AllowedVoiceList_en=["Justin","Salli","Ivy","Kendra","Matthew","Joanna",
    "Russell", "Nicole", "Amy", "Brian", "Emma", "Aditi", "Raveena",
                        "Aria", "Joey", "Kimberly"]

    mapping = {"PHOENIX":'Russell',
"EDGEWORTH":'Geraint',
"GODOT":'Joey',
"FRANZISKA":'Salli',
"JUDGE":'Brian',
"LARRY":'Joey',
"MAYA":'Raveena',
"KARMA":'Brian',
"PAYNE":'Geraint',
"MAGGEY":'Aditi',
"PEARL":'Joanna',
"LOTTA":'Emma',
"GUMSHOE":'Matthew',
"GROSSBERG":'Matthew',
"APOLLO":'Russell',
"KLAVIER":'Brian',
"MIA":'Aria',
"WILL":'Russell',
"OLDBAG":'Ayanda',
"REDD":'Russell'}
    if character_name not in character_idlist:
        character_idlist.append(character_name)
    # userid=character_idlist.index(character_name)   
    accent_name=mapping[character_name.upper()]
    if not accent_name or accent_name=="":
        accent_name = 'Brian'
    form_data = {
        "msg": text,
        "lang": accent_name,
        "source": "ttsmp3"
    }
    r = requests.post(TTSMP3_URL, form_data, proxies=proxies)
    if r.status_code == 200:
        # print(r.status_code)
        json = r.json()
        # print(json)
        try:
            url = json["URL"]
            filename = json["MP3"]
            mp3_file = requests.get(url, proxies=proxies)

            outputdir = ttsdir
            outputfilepath = outputdir+os.sep+str(index)+'.mp3'
            if os.path.exists(outputdir):

                print('sound directory exists', outputdir)
            else:
                os.makedirs(outputdir, exist_ok=True)
                print('create sound directory', outputdir)
            # print('tts', index, '----', outputfilepath)

            with open(outputfilepath, "wb") as out_file:
                out_file.write(mp3_file.content)

                print('ttsmp3 ok', outputfilepath)
            return outputfilepath
        # return mp3_file.content
        except:
            print('pls choose another tts tool')
            outputfilepath = synthentic_audio_ttstool(text, character_name,ttsdir,index)
            return outputfilepath   
    else:
        print('pls choose another tts tool')
        outputfilepath = synthentic_audio_ttstool(text, character_name,ttsdir,index)
        return outputfilepath
def synthentic_audio_ttstool(text, character_name,ttsdir,index):
    # dir = r'D:\Download\audio-visual\auto_video-jigsaw\tts\readaloud'
    try:
        web_driver = getwebdriver_chrome()
        
        url = 'https://ttstool.com/'
        out = web_driver.get(url)
        # print('========',out)
        # options = out.find_element_by_xpath("//select")
        # print("-----------\n",web_driver.page_source)

        wait = WebDriverWait(web_driver, 10)
        wait.until(EC.visibility_of_element_located(
            (By.XPATH, "/html/body/div[1]/table[1]/tbody/tr[2]/td[2]/select/option[4]")))
        while not web_driver.find_element_by_xpath("//table[1]/tbody/tr[1]/td[2]/select"):
            print('page not show')
            time.sleep(0.1)
            finished = web_driver.find_element_by_xpath(
                "//table[1]/tbody/tr[1]/td[2]/select")
            if finished == True:
                break

        select_tool = Select(web_driver.find_element_by_xpath(
            "//table[1]/tbody/tr[1]/td[2]/select"))

        # select by visible text
        select_tool.select_by_visible_text('Amazon')

        while not web_driver.find_element_by_xpath("//table[1]/tbody/tr[2]/td[2]/select") :
                print('page not show correctly')
                time.sleep(0.1)
                finished= web_driver.find_element_by_xpath("//table[1]/tbody/tr[2]/td[2]/select")
                if finished ==True:
                    break

        time.sleep(0.2)
        select_language = Select(web_driver.find_element_by_xpath(
            " /html/body/div[1]/table[1]/tbody/tr[2]/td[2]/select"))
        # /html/body/div[1]/table[1]/tbody/tr[2]/td[2]/select
        # print('11111',select_language)
        # select_language.select_by_visible_text('English')
        select_language.select_by_value('English')
        wait = WebDriverWait(web_driver, 10)
        download = wait.until(EC.visibility_of_element_located(
            (By.XPATH, '/html/body/div[1]/div/div[1]/i[1]')))
        # print('2222',download)
        character_idlist=[]
        AllowedVoiceList_en=["Justin","Salli","Ivy","Kendra","Matthew","Joanna",
        "Russell", "Nicole", "Amy", "Brian", "Emma", "Aditi", "Raveena",
                            "Aria", "Joey", "Kimberly"]

        mapping = {"PHOENIX":'Russell',
    "EDGEWORTH":'Geraint',
    "GODOT":'Joey',
    "FRANZISKA":'Salli',
    "JUDGE":'Brian',
    "LARRY":'Joey',
    "MAYA":'Raveena',
    "KARMA":'Brian',
    "PAYNE":'Geraint',
    "MAGGEY":'Aditi',
    "PEARL":'Joanna',
    "LOTTA":'Emma',
    "GUMSHOE":'Matthew',
    "GROSSBERG":'Matthew',
    "APOLLO":'Russell',
    "KLAVIER":'Brian',
    "MIA":'Aria',
    "WILL":'Russell',
    "OLDBAG":'Ayanda',
    "REDD":'Russell'}
        options ={"Nicole":"0",
            "Russell":"1",
            "Amy":"2",
            "Brian":"3",
            "Emma":"4",
            "Raveena":"5",
            "Ivy":"6",
            "Joey":"7",
            "Justin":"8",
            "Kendra":"9",
            "Kimberly":"10",
            "Salli":"11",
            "Geraint":"12"}
        if character_name not in character_idlist:
            character_idlist.append(character_name)
        accent_name=mapping[character_name.upper()]
        if accent_name in options:
            option="3"
        else:
            option=options[accent_name] 

        if download:
            select_voice = Select(web_driver.find_element_by_xpath(
                "//table[2]/tbody/tr[1]/td/div[2]/div[2]/select[1]"))

            select_voice.select_by_value(option)
            # print('146,',select_voice)
        inputElement = web_driver.find_element_by_xpath(
            "//table[2]/tbody/tr[1]/td/div[2]/div[3]/textarea")
        text = TextCorrectorbeforeTTS(text)
        inputElement.send_keys(text)
        ttsdirabsolute = os.getcwd()+os.sep+ttsdir
        tmp_outputdir = ttsdirabsolute+os.sep+'tmp'
        # print('-----',ttsdir)
        if os.path.exists(tmp_outputdir):

            # print('tmp sound directory exists',tmp_outputdir)

            files = glob.glob(tmp_outputdir+os.sep+'*')
            if files:
                for f in files:
                    print('remove tmp previous files', f)
                    os.remove(f)
        else:
            os.makedirs(tmp_outputdir, exist_ok=True)
        print('create tmp sound directory',tmp_outputdir)
        # Initializing the Chrome webdriver with the options

        # Setting Chrome to trust downloads
        web_driver.command_executor._commands["send_command"] = (
            "POST", '/session/$sessionId/chromium/send_command')
        params = {'cmd': 'Page.setDownloadBehavior', 'params': {
            'behavior': 'allow', 'downloadPath': tmp_outputdir}}
        command_result = web_driver.execute("send_command", params)
        print('----\n',command_result)

        # Click on the button and wait for 10 seconds

        web_driver.find_element_by_xpath(
            "/html/body/div[1]/div/div[1]/i[2]").click()
        time.sleep(2)
        # outputdir = ttsdirabsolute+os.sep+str(index)
        # if os.path.exists(outputdir):
        #     pass
        #     # print('sound directory exists',outputdir)
        # else:
        #     os.makedirs(outputdir, exist_ok=True)
        #     # print('create sound directory',outputdir)
        # # print('chongmingming')
        newfile =tiny_file_rename(str(index)+'.mp3', tmp_outputdir)
        web_driver.close()
        web_driver.quit()
        return newfile
    except:
        print('plan b tts service')
        # outputfilepath =ttsmp3_polly_tts(text, character_name,ttsdir,index)
        # return outputfilepath
# synthentic_audio_ttstool('','',1,dir)

# real person syn
# https://vo.codes/

def waitFor(maxSecond, runFunction, param):
    while maxSecond:
        try:
            return runFunction(param)
        except:
            time.sleep(0.5)
            maxSecond -= 0.5