import pandas as pd
import random


def pick_incorrect_authors(correct_author: str, data: pd.DataFrame):
    """
    Input: The dataframe and correct author
    Output: 3 incorrect authors
    """

    incorrect_authors = []

    while len(incorrect_authors) < 3:
        row = random.choice(range(len(data)))

        authors = str(data["authors"][row])
        authors = authors.split(";")

        distractor = authors[random.choice(range(len(authors)))]

        if distractor == correct_author:
            continue

        if distractor in incorrect_authors:
            continue

        incorrect_authors.append(distractor)

    return incorrect_authors


def pick_recent_articles(data: pd.DataFrame):
    """
    Returns a list of 4 articles one of which is clearly the most recent one.
    This will be the first one.
    """

    while True:
        article_rows = random.sample(range(len(data)), 4)
        articles = {}

        for row in article_rows:
            articles[data["title"][row]] = data["date_published"][row]

        articles = dict(
            sorted(articles.items(), key=lambda item: item[1], reverse=True)
        )

        latest_date = list(articles.values())[0]
        other_dates = list(articles.values())[1:4]

        if latest_date not in other_dates:
            break

    return list(articles)


path = "publications/publications/kmanpub.csv"
data = pd.read_csv(path)

row = random.choice(range(len(data)))


multiple_choice = pd.DataFrame(
    columns=["Question", "A", "B", "C", "D", "Correct"], dtype=str
)

# First author
for _ in range(20):
    row = random.choice(range(len(data)))

    title = data["title"][row]

    authors = str(data["authors"][row])
    authors = authors.split(";")

    question = "Who is the first author of the following article: " + title

    # Randomly decide which is going to be the correct anwser
    # Always index 0, whatever lands there
    options = ["A", "B", "C", "D"]
    options = random.sample(options, 4)
    correct = [options[0], authors[0]]

    # Pick random incorrect answers and assign them to the remaining indices
    incorrect_authors = pick_incorrect_authors(correct_author=correct[1], data=data)
    distractor1 = [options[1], incorrect_authors[0]]
    distractor2 = [options[2], incorrect_authors[1]]
    distractor3 = [options[3], incorrect_authors[2]]

    # write this information into the table
    new_row = len(multiple_choice)
    multiple_choice.loc[new_row] = str()
    multiple_choice.loc[new_row, "Question"] = question
    multiple_choice.loc[new_row, correct[0]] = correct[1]
    multiple_choice.loc[new_row, distractor1[0]] = distractor1[1]
    multiple_choice.loc[new_row, distractor2[0]] = distractor2[1]
    multiple_choice.loc[new_row, distractor3[0]] = distractor3[1]
    multiple_choice.loc[new_row, "Correct"] = correct[0]


# Which of these articles is the most recent?

for _ in range(20):
    # The first one is the most recent one
    titles = pick_recent_articles(data)
    question = "Which of these articles is the most recent one?"

    # Randomly decide which is going to be the correct anwser
    # Always index 0, whatever lands there
    options = ["A", "B", "C", "D"]
    options = random.sample(options, 4)
    correct = [options[0], titles[0]]

    distractor1 = [options[1], titles[1]]
    distractor2 = [options[2], titles[2]]
    distractor3 = [options[3], titles[3]]

    # write this information into the table
    new_row = len(multiple_choice)
    multiple_choice.loc[new_row] = str()
    multiple_choice.loc[new_row, "Question"] = question
    multiple_choice.loc[new_row, correct[0]] = correct[1]
    multiple_choice.loc[new_row, distractor1[0]] = distractor1[1]
    multiple_choice.loc[new_row, distractor2[0]] = distractor2[1]
    multiple_choice.loc[new_row, distractor3[0]] = distractor3[1]
    multiple_choice.loc[new_row, "Correct"] = correct[0]


multiple_choice.to_csv("multiple_coice.csv", index=False)
