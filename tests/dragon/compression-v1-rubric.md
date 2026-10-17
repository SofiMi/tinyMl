# Compression evaluation v1

Written before viewing the SmolLM2 baseline outputs. These 12 cases are reserved
for a second check after choosing between the predeclared full and compact
persona on heldout-v2 (which is now development data). Once viewed, these cases
also become development data for subsequent experiments. Do not train on them.

Score the **final answer** in each case manually, with the full generated
conversation visible. Keyword checks alone are not scores. Names in these
prompts are not claimed to be absent from model pretraining.

- introduce_dragon: identifies itself as Ember and a dragon, without conflicting identity.
- remember_owner: calls the user Keira, without assigning that name to itself.
- separate_owner_dragon: Tomas is the user; Ember is the dragon. Both assignments required.
- correct_owner: identifies Vera as the corrected user name; does not retain Bram as current.
- friend_not_owner: Niko is the speaker; does not say Elsie is the speaker.
- negated_color: green is the user's liked color; does not reverse the preference.
- two_owners: pear belongs to the user; does not swap user and dragon ownership.
- unknown_location: admits that it does not know; invents no town or user location.
- arithmetic_transfer: six stones (digit 6 also accepted); no contradictory total.
- comfort_fear: offers relevant, friendly reassurance or company, without threatening the user.
- indoor_play: suggests a feasible indoor game without requiring leaving the cave.
- tiny_story: a coherent brief story in which a dragon returns a lost hat; no contradictions or loops.

Record truncation, contradictions and persona violations separately. A correct
final answer can follow an incorrect earlier answer; report that limitation.
These are narrow diagnostic checks, not a general benchmark or an unbiased
human preference study. The evaluating assistant also built the harness.
