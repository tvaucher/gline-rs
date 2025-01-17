#place(
  top + center, float: true, scope: "parent",
  text(1.8em, weight: "bold")[Matrix-Level Documentation of Processing Steps]  
)

#place(
  top + center, float: true, scope: "parent",
  text[WORK IN PROGRESS]  
)

= Pre-Processing (Common)

== Text Input

This is the "user" input for the whole processing.

=== Source Code

- Related struct: `gliner::model::input::text::TextInput`

=== Format

- $n$: number of input texts
- $k$: number of entity class labels
- $I$: sequence of input texts matrix of type `string` and size $n$
- $E$: entity class labels matrix, of type `string` and size $k$

#align(left, block($ 
I =
  mat(delim: "[",
  "text"_1;
  "text"_2;
  dots.v;
  "text"_n
)
$))

#align(left, block($
E = mat(delim: "[",
  "label"_1;
  "label"_2;
  dots.v;
  "label"_k
)
$))


=== Example

#align(left, block($
I =
  mat(delim: "[",
  &"\"My name is James Bond\"";
  &"\"I like to drive my Aston Martin\"";  
)
$))

#align(left, block($
E = mat(delim: "[",
  &"\"movie character\"";
  &"\"vehicle\"";
)
$))

#pagebreak()
== Word-Level Tokenization

=== Transformation

#align(left, block($(I, E) -> (T, E)$))

=== Source Code

- Struct: `gliner::model::input::tokenized::TokenizedInput`
- Transformation: `gliner::model::input::prompt::RawToTokenized`

=== Format

- $n, k$:  same as before
- $T$: sequence of sequence of tokenized input texts, of type `string` and size $n$
- $E$: same as before

#align(left, block($
T = mat(delim: "[",
  mat(delim: "[", "token"_"1,1", "token"_"1,2", dots);
  mat(delim: "[", "token"_"2,1", "token"_"2,2", dots);
  dots.v;
  mat(delim: "[", "token"_"n,1", "token"_"n,2", dots);
)
$))

=== Example

#align(left, block($
T = mat(delim: "[",
  &mat(delim: "[", "\"My\"" "\"name\"", "\"is\"", "\"James\"", "\"Bond\"");  
  &mat(delim: "[", "\"I\"" "\"like\"", "\"to\"", "\"drive\"", "\"my\"", "\"Aston\"", "\"Martin\"");  
)
$))

#pagebreak()
== Prompt Preparation

Prepared prompts, appending entity and text tokens.

=== Transformation

#align(left, block($(T, E) -> P$))

=== Source Code

- Struct: `gliner::model::input::prompt::PromptInput`
- Transformation from `TokenizedInput`: `gliner::model::input::prompt::TokenizedToPrompt`

=== Format

#align(left, block($
P = mat(delim: "[",
  mat(delim: "[", "<<ENT>>", "label"_"1,1", "<<ENT>>", "label"_"1,2", dots, "<<SEP>>", "token"_"1,1", , "token"_"1,2", dots);
  mat(delim: "[", "<<ENT>>", "label"_"2,1", "<<ENT>>", "label"_"2,2", dots, "<<SEP>>", "token"_"2,1", , "token"_"2,2", dots);
  dots.v ;
  mat(delim: "[", "<<ENT>>", "label"_"n,1", "<<ENT>>", "label"_"n,2", dots, "<<SEP>>", "token"_"n,1", , "token"_"n,2", dots)
)
$))

=== Example

#align(left, block($
P = mat(delim: "[",
  &mat(delim: "[", "<<ENT>>", "\"movie character\"", "<<ENT>>", "\"vehicle\"", dots, "<<SEP>>", "\"My\"", "\"name\"", "\"is\"", "\"James\"", "\"Bond\"");
  &mat(delim: "[", "<<ENT>>", "\"movie character\"", "<<ENT>>", "\"vehicle\"", dots, "<<SEP>>", "\"I\"", "\"like\"", "\"to\"", "\"drive\"", "\"my\"", "\"Austin\"", "\"Martin\"");
)
$))


#pagebreak()
== Prompt Encoding (Sub-Word Tokenization)

=== Transformation

#align(left, block($P -> (I, M, W, L)$))

=== Source Code

- Struct: `gliner::model::input::encoded::EncodedPrompt`
- Transformation: `gliner::model::input::encoded::PromptsToEncoded`

=== Format

#let ststart(x) = text(fill: green, $#x$)
#let stend(x) = text(fill: red, $#x$)
#let stent(x) = text(fill: orange, $#x$)

- k: maximum number of sub-word tokens within a sequence, adding start ($ststart(1)$) and end ($stend(2)$) tokens
- I: encoded prompts of type `i64` and shape $(n*k)$
- A: attention masks of type `i64` and shape $(n*k)$
- W: word masks of type `i64` and shape $(n*k)$
- L: text lengths of type `i64` and shape $(n*1)$

#align(left, block($
I = mat(
  "token_id"_"1,1", "token_id"_"1,2", dots, "token_id"_"1,k" ;
  "token_id"_"2,1", "token_id"_"2,2", dots, "token_id"_"2,k" ;
  dots.v, dots.v, dots.down, dots.v ;
  "token_id"_"n,1", "token_id"_"n,2", dots, "token_id"_"n,k" ;  
)
$))

#align(left, block(
$
M = mat(
  "mask"_"1,1", "mask"_"1,2", dots, "mask"_"1,k" ;
  "mask"_"2,1", "mask"_"2,2", dots, "mask"_"2,k" ;
  dots.v, dots.v, dots.down, dots.v ;
  "mask"_"n,1", "mask"_"n,2", dots, "mask"_"n,k" ;  
)
$))

#align(left, block($
W = mat(
  "word_mask"_"1,1", "word_mask"_"1,2", dots, "word_mask"_"1,k" ;
  "word_mask"_"2,1", "word_mask"_"2,2", dots, "word_mask"_"2,k" ;
  dots.v, dots.v, dots.down, dots.v ;
  "word_mask"_"n,1", "word_mask"_"n,2", dots, "word_mask"_"n,k" ;  
)
$))

#align(left, block($
L = mat(
  "l"_"1"; 
  dots.v;
  "l"_"n";
)
$))

=== Example

#align(left, block($
I = mat(
  ststart(1), stent(128002), 1421, 1470, stent(128002), 1508, stent(128003), 573, 601, 269, 1749, 8728, stend(2), 0, 0;
  ststart(1), stent(128002), 1421, 1470, stent(128002), 1508, stent(128003), 273, 334, 264, 1168, 312, 20844, 2963, stend(2);
)
$))

#align(left, block($
M = mat(
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0;
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
)
$))

#align(left, block($
W = mat(
  0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0;
  0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0;
)
$))

#align(left, block($
L = mat(
  5;
  7
)
$))

#pagebreak()
= Pre-Processing (Span Mode)

Downstream of the aforementioned steps.

== Span Preparation

=== Transformation

#align(left, block($(I, M, W, L) -> (I, M, W, L, S_I, S_M)$))

=== Format

- $n, k, I, A, W, L$: same as before.
- $s$: maximum possible number of spans for one sequence
- $S_I$: span offsets, of type `i64` and shape $(n*s*2)$
- $S_M$: span masks, of type `bool` and shape $(n*s)$

#align(left, block($
S_I = mat(
  mat("start"_"1,1", "end"_"1,1"), mat("start"_"1,2", "end"_"1,2"), dots, mat("start"_"1,s", "end"_"1,s");
  mat("start"_"2,1", "end"_"2,1"), mat("start"_"2,2", "end"_"2,2"), dots, mat("start"_"2,s", "end"_"2,s");
  dots.v, dots.v, dots.down, dots.v;
  mat("start"_"n,1", "end"_"n,1"), mat("start"_"n,2", "end"_"n,2"), dots, mat("start"_"n,s", "end"_"n,s");
)
$))

#align(left, block($
S_M = mat(
  "span_mask"_"1,1", "span_mask"_"1,2", dots, "span_mask"_"1,s";
  "span_mask"_"2,1", "span_mask"_"2,2", dots, "span_mask"_"2,s";
  dots.v, dots.v, dots.down, dots.v;
  "span_mask"_"n,1", "span_mask"_"n,2", dots, "span_mask"_"n,s";
)
$))

=== Example

Note: for readability purposes, inside matrices are split into rows (one per token) but they are actually in one dimension $s$ (see format above).

#align(left, block($
S_I = mat(
  mat(
    mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3), mat(0, 4), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), ⏎; 
    mat(1, 1), mat(1, 2), mat(1, 3), mat(1, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(2, 2), mat(2, 3), mat(2, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(3, 3), mat(3, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(4, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, dots
  )
  ;
  mat(
    mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3), mat(0, 4), mat(0, 5), mat(0, 6), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), ⏎;
    mat(1, 1), mat(1, 2), mat(1, 3), mat(1, 4), mat(1, 5), mat(1, 6), mat(0, 0), dots, dots, dots, dots, dots, ⏎;
    mat(2, 2), mat(2, 3), mat(2, 4), mat(2, 5), mat(2, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, ⏎;
    mat(3, 3), mat(3, 4), mat(3, 5), mat(3, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(4, 4), mat(4, 5), mat(4, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(5, 5), mat(5, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(6, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots
  )
)
$))

#align(left, block($
S_M = mat(
  mat(
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ⏎; 
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  )
  ;
  mat(
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; 
  )
)
$))

= Pre-Processing (Token Mode)

Nothing more to be done beside the common steps.

#pagebreak()
= Post-Processing (Span Mode)

TODO

#pagebreak()
= Post-Processing (Token Mode)

TODO


