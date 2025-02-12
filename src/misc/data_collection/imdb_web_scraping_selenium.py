from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")  # Open browser in maximized mode
options.add_argument("--disable-extensions")  # Disable extensions
options.add_argument("--disable-gpu")  # Disable GPU for headless mode (optional)
options.add_argument("--no-sandbox")  # Bypass OS security model (Linux-specific)
options.add_argument("--disable-dev-shm-usage")  # Overcome resource problems (Linux-specific)
# options.add_argument("--headless")

# Initialize the Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
ratings = pd.read_csv('movie_ratings.csv', index_col=0)
for movie_id in ratings['tconst'][441:]:
  url = f"https://www.imdb.com/title/{movie_id}/reviews/"
  driver.get(url)
  # Allow the page to load
  time.sleep(3)  # Wait for JavaScript to render

  # Scroll down to load more reviews if necessary
  body = driver.find_element(By.CSS_SELECTOR, 'body')
  for _ in range(3):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(2)

  # Find all reviews on the page
  user_reviews_div = driver.find_elements(By.CLASS_NAME, "ipc-html-content-inner-div")
  user_ratings_div = driver.find_elements(By.CLASS_NAME, "ipc-rating-star--rating")
  user_ids_div = driver.find_elements(By.XPATH, "//a[contains(@href, '/user/ur')]")
  user_reviews = [user_review.text for user_review in user_reviews_div]
  user_ratings = [user_rating.text for user_rating in user_ratings_div]
  user_ids = [user_id.get_attribute('href').split("/")[4] for user_id in user_ids_div]
  min_length = min(len(user_ids), len(user_ratings), len(user_reviews))
  pd.DataFrame(
      {'userid': user_ids[0:min_length],
       'movie_id': min_length * [movie_id],
       'ratings': user_ratings[0:min_length],
       'reviews': user_reviews[0:min_length]}).to_csv('../raw_datasets/moview_user_reviews.csv', mode='a',
                                                      header=False,
                                                      index=False)

# Close the driver
driver.quit()
