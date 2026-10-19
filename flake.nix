{
  description = "Book RAG chatbot backend (FastAPI)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python pkgs.uv ];

          env = {
            # Make uv use the Nix-provided interpreter instead of downloading one.
            UV_PYTHON = "${python}/bin/python3";
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            uv sync --quiet
            source .venv/bin/activate
          '';
        };
      });
}
