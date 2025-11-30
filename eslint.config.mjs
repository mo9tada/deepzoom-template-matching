import pluginNext from "@next/eslint-plugin-next";

export default [
  pluginNext.configs["core-web-vitals"],
  {
    rules: {},
    ignores: ["node_modules/**", ".next/**", "out/**", "build/**", "next-env.d.ts"],
  },
];
