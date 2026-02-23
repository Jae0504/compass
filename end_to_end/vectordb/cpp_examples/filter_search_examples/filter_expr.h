#pragma once

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace filter_expr {

enum class LogicalOp {
    And,
    Or
};

enum class CompareOp {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le
};

struct Literal {
    std::string text;
    bool is_number = false;
    double number = 0.0;
};

struct Node {
    enum class Kind {
        Logical,
        Compare,
        In,
        Between
    };

    Kind kind = Kind::Compare;
    LogicalOp logical_op = LogicalOp::And;
    CompareOp compare_op = CompareOp::Eq;
    std::string field;
    Literal literal;
    Literal lower;
    Literal upper;
    std::vector<Literal> list;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

namespace detail {

enum class TokenType {
    Identifier,
    Number,
    String,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    KeywordAnd,
    KeywordOr,
    KeywordIn,
    KeywordBetween,
    End
};

struct Token {
    TokenType type = TokenType::End;
    std::string text;
    size_t pos = 0;
};

inline std::string uppercase_copy(std::string_view in) {
    std::string out(in.begin(), in.end());
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return out;
}

inline bool try_parse_double(std::string_view text, double* out) {
    if (text.empty()) {
        return false;
    }
    std::string copy(text);
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(copy.c_str(), &end);
    if (errno != 0 || end == copy.c_str() || *end != '\0' || !std::isfinite(v)) {
        return false;
    }
    if (out != nullptr) {
        *out = v;
    }
    return true;
}

inline std::vector<Token> tokenize(std::string_view expr) {
    std::vector<Token> tokens;
    size_t i = 0;
    while (i < expr.size()) {
        const char c = expr[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }

        if (c == '(') {
            tokens.push_back(Token{TokenType::LParen, "(", i++});
            continue;
        }
        if (c == ')') {
            tokens.push_back(Token{TokenType::RParen, ")", i++});
            continue;
        }
        if (c == '[') {
            tokens.push_back(Token{TokenType::LBracket, "[", i++});
            continue;
        }
        if (c == ']') {
            tokens.push_back(Token{TokenType::RBracket, "]", i++});
            continue;
        }
        if (c == ',') {
            tokens.push_back(Token{TokenType::Comma, ",", i++});
            continue;
        }

        if (i + 1 < expr.size()) {
            const std::string two(expr.substr(i, 2));
            if (two == "==") {
                tokens.push_back(Token{TokenType::Eq, two, i});
                i += 2;
                continue;
            }
            if (two == "!=") {
                tokens.push_back(Token{TokenType::Ne, two, i});
                i += 2;
                continue;
            }
            if (two == ">=") {
                tokens.push_back(Token{TokenType::Ge, two, i});
                i += 2;
                continue;
            }
            if (two == "<=") {
                tokens.push_back(Token{TokenType::Le, two, i});
                i += 2;
                continue;
            }
        }
        if (c == '>') {
            tokens.push_back(Token{TokenType::Gt, ">", i++});
            continue;
        }
        if (c == '<') {
            tokens.push_back(Token{TokenType::Lt, "<", i++});
            continue;
        }

        if (c == '\'' || c == '"') {
            const char quote = c;
            const size_t start = i;
            ++i;
            std::string s;
            while (i < expr.size()) {
                const char ch = expr[i];
                if (ch == '\\') {
                    if (i + 1 >= expr.size()) {
                        throw std::runtime_error("Invalid escape at end of string literal");
                    }
                    s.push_back(expr[i + 1]);
                    i += 2;
                    continue;
                }
                if (ch == quote) {
                    ++i;
                    break;
                }
                s.push_back(ch);
                ++i;
            }
            if (i > expr.size() || (i == expr.size() && expr[i - 1] != quote)) {
                throw std::runtime_error("Unterminated string literal at position " + std::to_string(start));
            }
            tokens.push_back(Token{TokenType::String, std::move(s), start});
            continue;
        }

        const bool maybe_signed_number =
            (c == '+' || c == '-') && (i + 1 < expr.size()) &&
            (std::isdigit(static_cast<unsigned char>(expr[i + 1])) || expr[i + 1] == '.');
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || maybe_signed_number) {
            const size_t start = i;
            if (c == '+' || c == '-') {
                ++i;
            }
            bool seen_digit = false;
            while (i < expr.size() && std::isdigit(static_cast<unsigned char>(expr[i]))) {
                seen_digit = true;
                ++i;
            }
            if (i < expr.size() && expr[i] == '.') {
                ++i;
                while (i < expr.size() && std::isdigit(static_cast<unsigned char>(expr[i]))) {
                    seen_digit = true;
                    ++i;
                }
            }
            if (!seen_digit) {
                throw std::runtime_error("Invalid numeric literal near position " + std::to_string(start));
            }
            if (i < expr.size() && (expr[i] == 'e' || expr[i] == 'E')) {
                const size_t exp_pos = i++;
                if (i < expr.size() && (expr[i] == '+' || expr[i] == '-')) {
                    ++i;
                }
                bool exp_digits = false;
                while (i < expr.size() && std::isdigit(static_cast<unsigned char>(expr[i]))) {
                    exp_digits = true;
                    ++i;
                }
                if (!exp_digits) {
                    throw std::runtime_error("Invalid exponent near position " + std::to_string(exp_pos));
                }
            }
            const std::string num(expr.substr(start, i - start));
            tokens.push_back(Token{TokenType::Number, num, start});
            continue;
        }

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const size_t start = i;
            ++i;
            while (i < expr.size()) {
                const char ch = expr[i];
                if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '.' || ch == '-') {
                    ++i;
                } else {
                    break;
                }
            }
            std::string ident(expr.substr(start, i - start));
            const std::string upper = uppercase_copy(ident);
            if (upper == "AND") {
                tokens.push_back(Token{TokenType::KeywordAnd, ident, start});
            } else if (upper == "OR") {
                tokens.push_back(Token{TokenType::KeywordOr, ident, start});
            } else if (upper == "IN") {
                tokens.push_back(Token{TokenType::KeywordIn, ident, start});
            } else if (upper == "BETWEEN") {
                tokens.push_back(Token{TokenType::KeywordBetween, ident, start});
            } else {
                tokens.push_back(Token{TokenType::Identifier, ident, start});
            }
            continue;
        }

        throw std::runtime_error("Unexpected character '" + std::string(1, c) + "' at position " +
                                 std::to_string(i));
    }
    tokens.push_back(Token{TokenType::End, "", expr.size()});
    return tokens;
}

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    std::unique_ptr<Node> parse() {
        std::unique_ptr<Node> root = parse_or_expr();
        expect(TokenType::End, "Expected end of expression");
        return root;
    }

private:
    std::unique_ptr<Node> parse_or_expr() {
        std::unique_ptr<Node> lhs = parse_and_expr();
        while (match(TokenType::KeywordOr)) {
            std::unique_ptr<Node> rhs = parse_and_expr();
            auto n = std::make_unique<Node>();
            n->kind = Node::Kind::Logical;
            n->logical_op = LogicalOp::Or;
            n->left = std::move(lhs);
            n->right = std::move(rhs);
            lhs = std::move(n);
        }
        return lhs;
    }

    std::unique_ptr<Node> parse_and_expr() {
        std::unique_ptr<Node> lhs = parse_primary();
        while (match(TokenType::KeywordAnd)) {
            std::unique_ptr<Node> rhs = parse_primary();
            auto n = std::make_unique<Node>();
            n->kind = Node::Kind::Logical;
            n->logical_op = LogicalOp::And;
            n->left = std::move(lhs);
            n->right = std::move(rhs);
            lhs = std::move(n);
        }
        return lhs;
    }

    std::unique_ptr<Node> parse_primary() {
        if (match(TokenType::LParen)) {
            std::unique_ptr<Node> n = parse_or_expr();
            expect(TokenType::RParen, "Expected ')'");
            return n;
        }
        return parse_predicate();
    }

    std::unique_ptr<Node> parse_predicate() {
        const Token field_tok = expect(TokenType::Identifier, "Expected field identifier");
        const std::string field_name = field_tok.text;

        if (match(TokenType::KeywordBetween)) {
            Literal lo = parse_literal();
            expect(TokenType::KeywordAnd, "Expected AND in BETWEEN clause");
            Literal hi = parse_literal();

            auto n = std::make_unique<Node>();
            n->kind = Node::Kind::Between;
            n->field = field_name;
            n->lower = std::move(lo);
            n->upper = std::move(hi);
            return n;
        }

        if (match(TokenType::KeywordIn)) {
            expect(TokenType::LBracket, "Expected '[' after IN");
            std::vector<Literal> values;
            if (!check(TokenType::RBracket)) {
                values.push_back(parse_literal());
                while (match(TokenType::Comma)) {
                    values.push_back(parse_literal());
                }
            }
            expect(TokenType::RBracket, "Expected ']' to close IN list");
            if (values.empty()) {
                throw std::runtime_error("IN list cannot be empty");
            }

            auto n = std::make_unique<Node>();
            n->kind = Node::Kind::In;
            n->field = field_name;
            n->list = std::move(values);
            return n;
        }

        CompareOp op;
        if (match(TokenType::Eq)) {
            op = CompareOp::Eq;
        } else if (match(TokenType::Ne)) {
            op = CompareOp::Ne;
        } else if (match(TokenType::Gt)) {
            op = CompareOp::Gt;
        } else if (match(TokenType::Ge)) {
            op = CompareOp::Ge;
        } else if (match(TokenType::Lt)) {
            op = CompareOp::Lt;
        } else if (match(TokenType::Le)) {
            op = CompareOp::Le;
        } else {
            throw std::runtime_error("Expected comparator after field '" + field_name + "'");
        }

        Literal rhs = parse_literal();
        auto n = std::make_unique<Node>();
        n->kind = Node::Kind::Compare;
        n->field = field_name;
        n->compare_op = op;
        n->literal = std::move(rhs);
        return n;
    }

    Literal parse_literal() {
        const Token tok = peek();
        Literal lit;
        if (match(TokenType::Number)) {
            lit.text = tok.text;
            lit.is_number = true;
            lit.number = std::strtod(tok.text.c_str(), nullptr);
            return lit;
        }
        if (match(TokenType::String)) {
            lit.text = tok.text;
            lit.is_number = false;
            return lit;
        }
        if (match(TokenType::Identifier)) {
            lit.text = tok.text;
            lit.is_number = try_parse_double(lit.text, &lit.number);
            return lit;
        }
        throw std::runtime_error("Expected literal at position " + std::to_string(tok.pos));
    }

    bool check(TokenType t) const {
        return peek().type == t;
    }

    bool match(TokenType t) {
        if (!check(t)) {
            return false;
        }
        ++idx_;
        return true;
    }

    Token expect(TokenType t, const std::string& msg) {
        const Token tok = peek();
        if (tok.type != t) {
            throw std::runtime_error(msg + " near position " + std::to_string(tok.pos));
        }
        ++idx_;
        return tok;
    }

    Token peek() const {
        if (idx_ >= tokens_.size()) {
            return Token{TokenType::End, "", 0};
        }
        return tokens_[idx_];
    }

    std::vector<Token> tokens_;
    size_t idx_ = 0;
};

inline bool compare_numeric(double lhs, double rhs, CompareOp op) {
    switch (op) {
        case CompareOp::Eq:
            return lhs == rhs;
        case CompareOp::Ne:
            return lhs != rhs;
        case CompareOp::Gt:
            return lhs > rhs;
        case CompareOp::Ge:
            return lhs >= rhs;
        case CompareOp::Lt:
            return lhs < rhs;
        case CompareOp::Le:
            return lhs <= rhs;
    }
    return false;
}

inline bool compare_string(const std::string& lhs, const std::string& rhs, CompareOp op) {
    switch (op) {
        case CompareOp::Eq:
            return lhs == rhs;
        case CompareOp::Ne:
            return lhs != rhs;
        case CompareOp::Gt:
            return lhs > rhs;
        case CompareOp::Ge:
            return lhs >= rhs;
        case CompareOp::Lt:
            return lhs < rhs;
        case CompareOp::Le:
            return lhs <= rhs;
    }
    return false;
}

template <class Accessor>
bool evaluate_node(const Node* node, const Accessor& accessor) {
    if (node == nullptr) {
        return false;
    }

    if (node->kind == Node::Kind::Logical) {
        if (node->logical_op == LogicalOp::And) {
            return evaluate_node(node->left.get(), accessor) && evaluate_node(node->right.get(), accessor);
        }
        return evaluate_node(node->left.get(), accessor) || evaluate_node(node->right.get(), accessor);
    }

    const std::optional<std::string_view> field_val_sv = accessor(node->field);
    if (!field_val_sv.has_value()) {
        return false;
    }
    const std::string field_val(field_val_sv->begin(), field_val_sv->end());

    if (node->kind == Node::Kind::Compare) {
        double lhs_num = 0.0;
        bool lhs_is_num = try_parse_double(field_val, &lhs_num);
        if (lhs_is_num && node->literal.is_number) {
            return compare_numeric(lhs_num, node->literal.number, node->compare_op);
        }
        return compare_string(field_val, node->literal.text, node->compare_op);
    }

    if (node->kind == Node::Kind::Between) {
        double lhs_num = 0.0;
        bool lhs_is_num = try_parse_double(field_val, &lhs_num);
        if (lhs_is_num && node->lower.is_number && node->upper.is_number) {
            return lhs_num >= node->lower.number && lhs_num <= node->upper.number;
        }
        return field_val >= node->lower.text && field_val <= node->upper.text;
    }

    if (node->kind == Node::Kind::In) {
        double lhs_num = 0.0;
        bool lhs_is_num = try_parse_double(field_val, &lhs_num);
        for (const Literal& lit : node->list) {
            if (lhs_is_num && lit.is_number) {
                if (lhs_num == lit.number) {
                    return true;
                }
            } else {
                if (field_val == lit.text) {
                    return true;
                }
            }
        }
        return false;
    }

    return false;
}

inline void collect_fields_impl(const Node* node, std::unordered_set<std::string>* out) {
    if (node == nullptr) {
        return;
    }
    if (node->kind == Node::Kind::Logical) {
        collect_fields_impl(node->left.get(), out);
        collect_fields_impl(node->right.get(), out);
        return;
    }
    out->insert(node->field);
}

}  // namespace detail

class Expression {
public:
    explicit Expression(std::string raw_expression) : source_(std::move(raw_expression)) {
        if (source_.empty()) {
            throw std::runtime_error("Filter expression cannot be empty");
        }
        std::vector<detail::Token> tokens = detail::tokenize(source_);
        detail::Parser parser(std::move(tokens));
        root_ = parser.parse();
    }

    const std::string& source() const {
        return source_;
    }

    template <class Accessor>
    bool evaluate(const Accessor& accessor) const {
        return detail::evaluate_node(root_.get(), accessor);
    }

    std::unordered_set<std::string> referenced_fields() const {
        std::unordered_set<std::string> out;
        detail::collect_fields_impl(root_.get(), &out);
        return out;
    }

private:
    std::string source_;
    std::unique_ptr<Node> root_;
};

}  // namespace filter_expr
