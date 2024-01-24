import time
import os
from io import BytesIO

import requests
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def download_image(url, folder_path, file_name):
  r = requests.get(url)
  if r.status_code == 200:
    image = Image.open(BytesIO(r.content))
    image.resize((72, 72)).save(os.path.join(folder_path, file_name))


mixmoji_data_path = "data/mixmoji"
if not os.path.exists(mixmoji_data_path):
  os.makedirs(mixmoji_data_path)
mixmoji_data_path = "data/mixmoji/mixmoji"
if not os.path.exists(mixmoji_data_path):
  os.makedirs(mixmoji_data_path)
emoji_data_path = "data/mixmoji/emoji"
if not os.path.exists(emoji_data_path):
  os.makedirs(emoji_data_path)

# Initialize WebDriver
driver = webdriver.Chrome('/usr/bin/chromedriver')

mixjojis = {}
for mixmoji in os.listdir(mixmoji_data_path):
  emoji1, emoji2 = os.path.splitext(mixmoji)[0].split('_')
  if emoji1 not in mixjojis:
    mixjojis[emoji1] = []
  mixjojis[emoji1].append(emoji2)

  if emoji2 not in mixjojis:
    mixjojis[emoji2] = []
  mixjojis[emoji2].append(emoji1)

try:
  driver.get("https://emoji.supply/kitchen/")
  time.sleep(2)

  emoji_buttons = driver.find_elements(By.CSS_SELECTOR, '#emoji-container #emoji-content div')
  for emoji_button in emoji_buttons:
    name = emoji_button.get_attribute('id')
    emoji_img = emoji_button.find_element(By.CSS_SELECTOR, 'img')
    link = emoji_img.get_attribute('src')
    download_image(link, emoji_data_path, f"{name}.png")

  for emoji_button in emoji_buttons:
    emoji_button.click()

    try:
      # Wait for the second container to be ready and find all buttons
      mixmoji_buttons = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '#mixmoji-container #mixmoji-content img'))
      )
    except TimeoutException as ex:
      print("No mixmoji container found for emoji")
      continue

    for mixmoji_button in mixmoji_buttons:
      # mixmoji_button.click()

      name = mixmoji_button.get_attribute('id')
      link = mixmoji_button.get_attribute('src')
      emoji1, emoji2 = name.split('_')
      if (emoji1 in mixjojis and emoji2 in mixjojis[emoji1]) or (emoji2 in mixjojis and emoji1 in mixjojis[emoji2]):
        continue
      download_image(link, mixmoji_data_path, f"{name}.png")

      mixjojis[emoji1].append(emoji2)
      mixjojis[emoji2].append(emoji1)

      # img_element = WebDriverWait(driver, 10).until(
      #   EC.presence_of_element_located((By.CSS_SELECTOR, '#preview-container #pc'))
      # )
      # link = img_element.get_attribute('src')
      # name = img_element.get_attribute('name')
      # download_image(link, mixmoji_data_path, f"{name}.png")

      time.sleep(0.5)

finally:
  driver.quit()
