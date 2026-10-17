#include "dragon.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

static void contains(const DragonReply& reply, const char* expected) {
    if (reply.text.find(expected) == std::string::npos) {
        std::cerr << "Expected '" << expected << "' in: " << reply.text << '\n';
        std::abort();
    }
}
static void unchanged(Dragon& dragon, const std::string& input) {
    auto before = dragon.memory();
    assert(!dragon.respond(input).memory_changed);
    assert(std::memcmp(&before, &dragon.memory(), sizeof(before)) == 0);
}
int main() {
    Dragon d;
    contains(d.respond("What is my name?"), "don't know");
    unchanged(d, "Thank you");
    assert(!*d.memory().owner);
    d.respond("Hello!");
    unchanged(d, "Tell me a story");
    assert(!*d.memory().owner);
    d.respond("Hello!");
    assert(d.respond("Sofia").memory_changed);
    assert(std::string(d.memory().owner) == "Sofia");
    unchanged(d, "My name is Sofia"); // no flash write for unchanged facts
    contains(d.respond("You are you?"), "I am Ember");
    for (const auto* text : {"I am hungry", "I am Sad", "I am not Sofia",
                            "My name is Alex?", "My name is \"Alex\"",
                            "You are Sofia", "I'm a dragon", "My name is Ember"}) unchanged(d, text);
    assert(std::string(d.memory().owner) == "Sofia");
    contains(d.respond("No, I am Alice"), "Hello, Alice");
    contains(d.respond("What are our names?"), "You are Alice");
    d.respond("My friend is Sofia");
    contains(d.respond("What is my friend's name?"), "Sofia");
    contains(d.respond("What is my name?"), "Alice");
    d.respond("I like green, not orange");
    unchanged(d, "I do not like blue");
    unchanged(d, "I like red|green");
    contains(d.respond("What color do I like?"), "green");
    d.respond("Actually, my favorite color is blue");
    contains(d.respond("What is my favorite color?"), "blue");
    d.respond("My favorite food is pears");
    contains(d.respond("What is my favorite food?"), "pears");
    contains(d.respond("What is your favorite food?"), "apples");
    contains(d.respond("Where do I live?"), "don't know");
    d.respond("I live in Moscow");
    contains(d.respond("Where do I live?"), "Moscow");
    d.respond("The hat is mine. The key is yours.");
    contains(d.respond("What belongs to me?"), "hat is yours");
    contains(d.respond("What belongs to you?"), "key is mine");
    d.respond("I gave you a blue stone");
    contains(d.respond("What color was my gift?"), "blue");
    d.respond("The blue stone is mine");
    assert(!*d.memory().dragon_item && !*d.memory().gift_color);
    contains(d.respond("What belongs to me?"), "blue stone");
    for (int a = -10; a <= 10; ++a) for (int b = -10; b <= 10; ++b) {
        for (char op : {'+', '-'}) {
            const auto input = std::to_string(a) + " " + op + " " + std::to_string(b) + "?";
            const auto before = d.memory();
            contains(d.respond(input), ("= " + std::to_string(op == '+' ? a+b : a-b) + ".").c_str());
            assert(std::memcmp(&before, &d.memory(), sizeof(before)) == 0);
        }
    }
    contains(d.respond("1 + 1?"), "= 2.");
    contains(d.respond("?"), "makes 2");
    auto clarification = d.respond("?").text;
    assert(clarification != d.respond("?").text);
    d.respond("Let's play");
    contains(d.respond("5"), "Five!");
    d.respond("Play a game");
    contains(d.respond("4"), "make five");
    const auto before = d.memory();
    d.new_conversation();
    assert(std::memcmp(&before, &d.memory(), sizeof(before)) == 0);
    unchanged(d, std::string(257, 'a'));
    unchanged(d, "Привет");
    for (int i = 0; i < 100; ++i) unchanged(d, "Tell me a story");
    contains(d.respond("What is my name?"), "Alice");
    Dragon restored;
    assert(restored.restore(d.memory()));
    contains(restored.respond("What is my name?"), "Alice");
    auto corrupt = d.memory();
    std::memset(corrupt.owner, 'x', sizeof(corrupt.owner));
    assert(!restored.restore(corrupt));
    corrupt = d.memory(); corrupt.version = 2;
    assert(!restored.restore(corrupt));
    d.forget();
    contains(d.respond("What is my name?"), "don't know");
    assert(!*d.memory().color && !*d.memory().friend_name);
    std::cout << "PASS: identity, corrections, ownership, unknown facts, reset, arithmetic, bounds\n";
}
