<!---
Blue Brain Search is a text mining toolbox focused on scientific use cases.

Copyright (C) 2020  Blue Brain Project, EPFL.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
-->

# Contributing
Thank you for your interest in contributing to Blue Brain Search!

Before you open an issue or start working on a pull request, please make sure
to read the following guidelines.

1. [Creating Issues](#CreatingIssues)
    1. [Search related issues](#Sri)
    1. [Open new issue](#Oni)
1.  [Creating Pull Requests](#CreatingPR)
    1. [Refer to an issue](#Ptai)
    1. [Add unit test](#Aui)
    1. [Add type annotations](#Ata)
    1. [Update dependencies](#Udep)
    1. [Update documentation](#Udoc)
    1. [Ensure all CI tests pass](#Ttc)
    1. [Get reviews and approval](#Ac)

## <a name="CreatingIssues"></a> Creating Issues
This section contains the instructions to be followed to create issues on our
[issue tracker](https://github.com/BlueBrain/Search/issues).

### <a name="Sri"></a> Search related issues
Before you submit an issue, please search our 
[issue tracker](https://github.com/BlueBrain/Search/issues) to verify 
if any similar issue was raised in the past. Even if you cannot find a direct
answer to your question, this can allow you to collect a list of related issues
that you can link in your new issue.

### <a name="Oni"></a> Open new issue
Once you are done with that, feel free to open a new issue. To make sure that 
your issue can be promptly addressed, please use one of the available templates
and try to fill every field in it. 

- **🐛 Bug Report** — You encountered some error or unexpected
  behavior.
- **🚀 Feature Request** — You would like some feature to be added or
  improved.
- **📚 Documentation** — You found something wrong or missing in the
  docs.
- **❓ Other Questions / Help** — For any other question or help
  request. 


## <a name="CreatingPR"></a> Creating Pull Requests
If you wish to contribute to the code base, opening a pull request on GitHub is
the right thing to do!
 
Please read the following paragraphs to make sure that your work can be
considered for a potential integration in our source code. 

### <a name="Ptai"></a> Refer to an issue
In general, every pull request should refer to a specific issue. If you would
like to provide your contribution on a untracked issue, please create first an 
issue as explained [here](#CreatingIssues) so that we can discuss the value of 
the proposed contribution and its implementation.

### <a name="Aui"></a> Add unit tests
Concerning CI tests, we are running various checks on linting, unit tests, docs,
and packaging. If you are adding or modifying a functionality in the code, you
are also expected to provide a unit test demonstrating and checking the new
behavior. 

### <a name="Ata"></a> Add type annotations
We are gradually introducing type annotations to our code, and our CI performs
type checking with [mypy](https://mypy.readthedocs.io/en/stable/index.html). If
your PR introduces a new function or heavily modifies an existing function, you
should also add type annotations for such function.   

### <a name="Udep"></a> Update dependencies
If your PR introduces a dependency on a new python package, this should be
added to both the `setup.py` and the `requirements.txt` files.

### <a name="Udoc"></a> Update documentation
The
[`whatsnew.rst`](https://github.com/BlueBrain/Search/blob/master/docs/source/whatsnew.rst)
file in our docs keeps tracks of the updates introduced in every new release,
so you should update it if your PR adds or changes a feature, fixes a bug, etc.

Moreover, instructions and examples in the docs should be updated whenever
deemed appropriate.  

### <a name="Ttc"></a> Ensure all CI tests pass
We use GitHub Actions to run [our CI tests](https://github.com/BlueBrain/Search/actions?query=workflow%3A%22ci+testing%22). 
Once you open a PR, the workflow that runs all CI tests is automatically 
triggered. This workflow is also triggered every time a new commit is pushed to 
that PR branch. If you want to push a commit without triggering the CI tests
(e.g. if a feature is still work in progress and you want to save time), your
 commit message should include an appropriate label like `[skip ci]`, `[ci skip]`, 
`[no ci]` etc. (see 
[here](https://github.blog/changelog/2021-02-08-github-actions-skip-pull-request-and-push-workflows-with-skip-ci/) 
for the complete list).

All CI tests must pass before the pull request can be considered for review.

### <a name="Ac"></a> Get reviews and approval
Once you have satisfied all the previous points, feel free to open your pull
request!

We will get back to you as soon as possible with comments and feedback in the
format of pull request reviews. At least two positive reviews from the
maintainers are required for the pull request to be merged into the master.


[ml-team-email]: mailto:bbp-ou-machinelearning@groupes.epfl.ch
[github]: https://github.com/BlueBrain/Search
