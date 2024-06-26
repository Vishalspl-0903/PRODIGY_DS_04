import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_data_path = '/content/twitter_training.csv'
validation_data_path = '/content/twitter_validation.csv'

training_df = pd.read_csv(training_data_path, header=None, names=['id', 'entity', 'sentiment', 'text'])
validation_df = pd.read_csv(validation_data_path, header=None, names=['id', 'entity', 'sentiment', 'text'])

df = pd.concat([training_df, validation_df])
print(df.head())

plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
entity_sentiment = df.groupby(['entity', 'sentiment']).size().unstack().fillna(0)
entity_sentiment.plot(kind='bar', stacked=True)
plt.title('Sentiment by Entity')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.show()
