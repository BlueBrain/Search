# Contributing

Thank you for your interest in contributing to Blue Brain Search!

Before you open an issue or start writing some new code, please make sure to 
read the following guidelines.


## <a name="NewIssues"></a> Creating New Issues

Before you submit an issue, please search our 
[issue tracker](https://github.com/BlueBrain/BlueBrainSearch/issues) to verify 
if any similar issue was raised in the past. Even if you cannot find a direct
answer to your question, this can allow you to collect a list of related issues
that you can link in your new issue.

Once you are done with that, feel free to open a new issue. To make sure that 
your issue can be promptly addressed, please use one of the available templates
and try to fill every field in it. 

- **🐛 Bug Report** — You encountered some error or unexpected behavior.
- **🚀 Feature Request** — You would like some feature to be added or improved.
- **📚 Documentation** — You found something wrong or missing in the docs.
- **❓ Other Questions / Help** — For any other question or help request. 


## Creating Pull Requests

If you wish to contribute to the code base, opening a pull request on GitHub is
the right thing to do. Please read the following paragraphs to make sure that
your work can be considered for a potential integration in our source code. 

In general, every pull request should refer to a specific issue. If you would
like to provide your contribution on a untracked issue, please open first a new 
issue as explained [here](#NewIssues) so that we can discuss the value of the
proposed contribution and its implementation.

Once you have an issue you can refer to, feel free to open a pull request. 
Before merging into the `master`, it is necessary that the two following
conditions are satisfied.
1. At least 2 maintainers provide their approval.
2. All CI tests pass.

Concerning CI test, we are running various checks on linting, unit tests, docs, 
and packaging. If you are adding or modifying a functionality in the code, you
are also expected to provide a unit test demonstrating and checking the new 
behavior. 

We are gradually introducing type annotations to our code, and our CI performs
type checking with [mypy](https://mypy.readthedocs.io/en/stable/index.html). If
your PR introduces a new function or heavily modifies an existing function, you
should also add type annotations for such function.   

To save time, by default our Travis CI pipeline does not test
commits on PR branches unless the commit message contains a `[test travis]` or 
`[run travis]` tag. So make sure to add such tag to your last commit (or add an 
empty commit with such tag) before requesting a review or your PR.


## Signing the CLA

If you are not part of the Blue Brain project, a Contributor License Agreement 
(CLA) must be signed for any code changes to be accepted. Please contact the 
[Blue Brain Project (EPFL) - Machine Learning Team][ml-team-email] to get the latest 
CLA version and instructions.

[ml-team-email]: mailto:bbp-ou-machinelearning@groupes.epfl.ch
[github]: https://github.com/BlueBrain/BlueBrainSearch
