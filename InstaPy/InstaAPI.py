from instapy import InstaPy
from instapy import smart_run

insta_username = 'miamcpea'
insta_password = 'Insta123@'

session = InstaPy(
	username=insta_username,
	password=insta_password,
	headless_browser=True)  # Permet de ne pas ouvrir de navigateur et fonctionner de manière invisible

with smart_run(session):
	# Permet de définir quel type de profil vous voulez suivre, s'ils ont beaucoup d'abonnés
	# session.set_relationship_bounds(
	enabled = True,
	delimit_by_numbers = True,
	max_followers = 5000,
	min_followers = 50,
	min_following = 100
