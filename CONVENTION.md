# Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) for every commit:

```text
<type>[optional scope]: <short imperative description>
```

Common types:

- `feat`: add user-facing functionality
- `fix`: correct a bug
- `docs`: documentation only
- `test`: add or change tests
- `refactor`: restructure code without changing behavior
- `chore`: maintenance or tooling

Use a concise, lowercase, kebab-case scope when it adds useful context, for
example `fix(data-loader): handle Excel dates`. Omit the scope when unnecessary.
Write the subject in imperative mood, without a trailing period.

For breaking changes, add `!` after the type or scope and describe the change
in the commit body or a `BREAKING CHANGE:` footer:

```text
feat(api)!: remove the legacy response format
```

Husky runs Commitlint on commit and rejects messages that do not follow the
configured Conventional Commits rules. Install the Git hook with `npm install`.
