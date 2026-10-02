/** Preset programs for the Prolog showcase: each with queries to try. */

export type Preset = { label: string; program: string; queries: readonly string[] }

export const PRESETS = {
  family: {
    label: 'family tree',
    program: `% Facts: who is whose parent.
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).
parent(pat, jim).

% A rule: X is a grandparent of Z if X is a parent
% of some Y, and that Y is a parent of Z.
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% Recursion: an ancestor is a parent, or a parent
% of an ancestor.
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
`,
    queries: ['grandparent(tom, Who)', 'ancestor(tom, D)', 'grandparent(G, jim)', 'ancestor(jim, X)'],
  },
  lists: {
    label: 'lists: append and reverse',
    program: `% app(A, B, C): the list C is A followed by B.
% Asked with C known, it splits C every way.
app([], L, L).
app([H|T], L, [H|R]) :- app(T, L, R).

% Naive reverse: reverse the tail, then append
% the head at the end.
rev([], []).
rev([H|T], R) :- rev(T, RT), app(RT, [H], R).
`,
    queries: ['app(X, Y, [1, 2, 3])', 'rev([1, 2, 3], R)', 'app([a, b], [c], L)', 'app(X, [3], [1, 2, 3])'],
  },
  control: {
    label: 'cut and negation',
    program: `% Cut (!) commits to the first clause once
% X >= Y holds: the second clause is not tried.
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).

% Negation as failure: \\+ G succeeds when G
% has no proof.
bird(tweety).
bird(pingu).
penguin(pingu).
flies(B) :- bird(B), \\+ penguin(B).
`,
    queries: ['max(7, 2, M)', 'max(3, 5, M)', 'flies(B)', 'bird(B), \\+ flies(B)'],
  },
  colouring: {
    label: 'map colouring',
    program: `% Colour four regions so that neighbours differ.
% a borders b and c; b borders c and d;
% c borders d.
colour(red).
colour(green).
colour(blue).

map(A, B, C, D) :-
  colour(A), colour(B), A \\= B,
  colour(C), A \\= C, B \\= C,
  colour(D), B \\= D, C \\= D.
`,
    queries: ['map(red, B, C, D)', 'map(A, B, C, D)', 'map(A, green, A, D)'],
  },
  einstein: {
    label: "Einstein's puzzle (zebra)",
    program: `% Einstein's puzzle. Five houses in a row, each with a colour, an owner's
% nationality, a drink, a cigar brand and a pet: h(Colour, Nation, Drink, Smoke, Pet).
% Who owns the fish?
right_of(X, Y, [Y, X | _]).
right_of(X, Y, [_ | T]) :- right_of(X, Y, T).
next_to(X, Y, L) :- right_of(X, Y, L).
next_to(X, Y, L) :- right_of(Y, X, L).

houses(Hs) :-
    Hs = [h(_, norwegian, _, _, _), _, h(_, _, milk, _, _), _, _],  % clues 9 and 8
    member(h(red, brit, _, _, _), Hs),                              % 1
    member(h(_, swede, _, _, dog), Hs),                             % 2
    member(h(_, dane, tea, _, _), Hs),                              % 3
    right_of(h(white, _, _, _, _), h(green, _, _, _, _), Hs),       % 4: green just left of white
    member(h(green, _, coffee, _, _), Hs),                          % 5
    member(h(_, _, _, pallmall, birds), Hs),                        % 6
    member(h(yellow, _, _, dunhill, _), Hs),                        % 7
    next_to(h(_, _, _, blends, _), h(_, _, _, _, cats), Hs),        % 10
    next_to(h(_, _, _, _, horse), h(_, _, _, dunhill, _), Hs),      % 11
    member(h(_, _, beer, bluemasters, _), Hs),                      % 12
    member(h(_, german, _, prince, _), Hs),                         % 13
    next_to(h(_, norwegian, _, _, _), h(blue, _, _, _, _), Hs),     % 14
    next_to(h(_, _, _, blends, _), h(_, _, water, _, _), Hs),       % 15
    member(h(_, _, _, _, fish), Hs).

fish_owner(Who) :- houses(Hs), member(h(_, Who, _, _, fish), Hs).
`,
    queries: ['fish_owner(Who)', 'houses(Hs)'],
  },
} satisfies Record<string, Preset>

export type PresetName = keyof typeof PRESETS
