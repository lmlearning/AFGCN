"""Dependency-free parser for ICCMA ``p af N`` argumentation frameworks."""


def read_af_input(file_path):
    arguments = None
    attacks = []
    with open(file_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            prefix = f"{file_path}:{line_number}: "
            if parts[0] == "p":
                if arguments is not None:
                    raise ValueError(prefix + "duplicate framework header")
                if len(parts) != 3 or parts[1] != "af":
                    raise ValueError(prefix + "expected 'p af N'")
                try:
                    count = int(parts[2])
                except ValueError:
                    raise ValueError(prefix + "argument count must be an integer") from None
                if count < 0:
                    raise ValueError(prefix + "argument count must be nonnegative")
                arguments = [str(i) for i in range(1, count + 1)]
                known_arguments = set(arguments)
            else:
                if arguments is None:
                    raise ValueError(prefix + "attack before framework header")
                if len(parts) != 2:
                    raise ValueError(prefix + "expected two attack endpoints")
                if any(endpoint not in known_arguments for endpoint in parts):
                    raise ValueError(prefix + "attack endpoint outside declared arguments")
                attacks.append(parts)
    if arguments is None:
        raise ValueError(f"{file_path}: missing 'p af N' header")
    return arguments, attacks
