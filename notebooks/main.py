import click
import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preprocess_text(text):
    return re.sub(r"[^\w\s]+", '', text).lower()


@click.group()
def main():
    pass


@main.command()
@click.option("--data", type=click.Path(exists=True))
@click.option("--test", type=click.Path())
@click.option("--split", type=float)
@click.option("--model", type=click.Path())
def train(data, test, split, model):
    df = pd.read_csv(data)
    df['rating_1_or_0'] = df['rating'].apply(lambda x: int(x > 3))
    df['report'] = df['title'] + '. ' + df['text']
    columns_to_drop = ['text', 'title', 'published_date', 'type', 'published_platform', 'rating', 'helpful_votes']
    df.drop(columns_to_drop, axis=1, inplace=True)
    df = df.dropna()
    df['report'] = df['report'].apply(preprocess_text)
    x = df['report']
    y = df['rating_1_or_0']
    if test:
        test_df = pd.read_csv(test)
        test_df['rating_1_or_0'] = test_df['rating'].apply(lambda x: int(x > 3))
        test_df['report'] = test_df['title'] + '. ' + test_df['text']
        test_df_columns_to_drop = ['text', 'title', 'published_date', 'type', 'published_platform', 'rating', 'helpful_votes']
        test_df.drop(columns_to_drop, axis=1, inplace=True)
        test_df = test_df.dropna()
        test_df['report'] = test_df['report'].apply(preprocess_text)
        x_train = x
        x_test = test_df['report']
        y_train = y
        y_test = test_df['rating_1_or_0']
    elif split:
        if not 0 <= split <= 1:
            raise click.BadParameter("Split value must be between 0 and 1")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
    elif model:
        x_train = x
        x_test = None
        y_train = y
        y_test = None
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(max_iter=200))
    ])
    my_model = pipeline.fit(x_train, y_train)
    if x_test is not None:
        y_pred = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        click.echo(f"Accuracy on test set: {accuracy}")

    with open(model, "wb") as f:
        pickle.dump(my_model, f)
    click.echo(f"Model saved to {model}")


@main.command()
@click.option("--model", type=click.Path())
@click.option("--data", type=click.Path())
def predict(model, data):
    with open(model, "rb") as f:
        my_model = pickle.load(f)
    if data.endswith(".csv"):
        df = pd.read_csv(data)
        df['report'] = df['title'] + '. ' + df['text']
        columns_to_drop = ['text', 'title', 'published_date', 'type', 'published_platform', 'rating', 'helpful_votes']
        df.drop(columns_to_drop, axis=1, inplace=True)
        df = df.dropna()
        df['report'] = df['report'].apply(preprocess_text)
        x = df['report']
        pred = my_model.predict(x)
        for i in pred:
          click.echo(i)
    else:
        pred = my_model.predict([data])
        click.echo(pred[0])


if __name__ == '__main__':
    main()
