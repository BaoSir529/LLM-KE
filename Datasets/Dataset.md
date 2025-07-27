## Dataset

In this paper, we reconstruct and publish the Twitter-V2. The main modifications are deleting incorrect aspect terms and sensitive data, adding missing aspect terms, and balancing the distribution of sentiment labels.

- 📚 Twitter-2015, Twitter-2017 and Twitter-V2 can be download from *Webdev* (Publication after review).

### Data structures

```json
{
	"img": "17_06_9405.jpg",
	"image_tweets_intensity": 85,
	"tweet": "NBA Finals : Is LeBron James facing his own basketball mortality after falling . . . -",
	"aspect": "LeBron James",
	"polarity": "NEG"
},
{
	"img": "1329448.jpg",
	"image_tweets_intensity": 85.0,
	"tweet": "Beautiful day today to have a walk in the Koutoubia Garden # Marrakech $T$ # Morroco",
	"aspect": "# Maroc",
	"polarity": "NEU"

},
{

	"img": "16_05_03_421.jpg",
	"image_tweets_intensity": 85,
	"tweet": "Fan grabs Thunder player ' s arm during wild finish # nba",
	"aspect": "nba",
	"polarity": "NEU"

}
```