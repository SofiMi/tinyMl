import unittest
from prepare_dialogue_v2 import select_conversation


def dialogue(topic):
    return [{"role":"user","content":"Hello!"}, {"role":"assistant","content":"Hello!"},
            {"role":"user","content":topic}, {"role":"assistant","content":"Let us talk about that."}]


class ConversationSelectionTests(unittest.TestCase):
    def test_same_greeting_does_not_merge_different_conversations(self):
        a = select_conversation(dialogue("What can we do on a rainy day?"))
        b = select_conversation(dialogue("How can I grow a small garden?"))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])
        self.assertEqual(a[2][0]["content"], "Hello!")

    def test_same_topic_stays_in_same_split_despite_greeting(self):
        a = dialogue("What can we do on a rainy day?")
        b = dialogue("What can we do on a rainy day?")
        b[0]["content"] = "Hi there!"
        self.assertEqual(select_conversation(a)[1], select_conversation(b)[1])
        self.assertNotEqual(select_conversation(a)[0], select_conversation(b)[0])


if __name__ == "__main__":
    unittest.main()
