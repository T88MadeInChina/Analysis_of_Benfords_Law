from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
REFERENZINTERVALL_FILE = 'cleaned_referenzintervall.csv'


def read_csv_file(file_path):
    sep_options = [',', ';', '\t']
    encoding_options = ['utf-8', 'latin1']
    df = None

    for encoding in encoding_options:
        for sep in sep_options:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                return df
            except (pd.errors.ParserError, UnicodeDecodeError):
                continue
    return df


def find_columns(df):
    age_columns = [col for col in df.columns if col.lower() in ['age', 'alter']]
    sex_columns = [col for col in df.columns if col.lower() in ['sex', 'geschlecht']]
    return age_columns, sex_columns


def get_column_range(df, column):
    return df[column].dropna().unique()


def leading_digit_analysis(data):
    MIN_NONZERO_VALUE = 1e-10
    data = data.dropna().astype(str)
    data = data[data.str.replace('.', '').str.isdigit()]
    leading_digits = data.str.split('.').str[0].str[0].astype(int)
    leading_digits = leading_digits[leading_digits != 0]  # 过滤掉首位数字为0的情况
    leading_digit_counts = leading_digits.value_counts().sort_index()
    total_count = leading_digits.count()
    leading_digit_freq = leading_digit_counts / total_count

    # 添加最小值逻辑
    for i in range(1, 10):
        if i not in leading_digit_freq:
            leading_digit_freq.loc[i] = MIN_NONZERO_VALUE
        elif leading_digit_freq.loc[i] == 0:
            leading_digit_freq.loc[i] = MIN_NONZERO_VALUE

    leading_digit_freq = leading_digit_freq.sort_index()
    return leading_digit_freq


def benford_distribution():
    return np.log10(1 + 1 / np.arange(1, 10))


def chi_square_test(observed, expected, total_count):
    chi2_stat = sum(((observed * total_count - expected * total_count) ** 2) / (expected * total_count))
    return chi2_stat


def z_test(observed, expected, total_count):
    pe = expected
    po = observed
    z_scores = []
    significant_digits = []
    for i in range(len(po)):
        pe_i = pe[i]
        po_i = po[i]
        z_score = (np.sqrt(total_count) * (abs(po_i - pe_i) - (1 / (2 * total_count)))) / np.sqrt(pe_i * (1 - pe_i))
        z_scores.append(z_score)
        if abs(z_score) > 1.96:
            significant_digits.append(i + 1)
    return z_scores, significant_digits


def measure_of_distance(observed, expected):
    return np.linalg.norm(observed - expected)


def regression_measure(observed):
    benford_proportions = np.log10(1 + 1 / np.arange(1, 10)).reshape(-1, 1)
    observed_proportions = observed.reshape(-1, 1)
    model = LinearRegression().fit(benford_proportions, observed_proportions)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    return slope, intercept


def list_uploaded_files():
    return os.listdir(UPLOAD_FOLDER)


@app.route('/', methods=['GET', 'POST'])
def index():
    selected_file = None
    context = {'files': list_uploaded_files()}
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            context['files'] = list_uploaded_files()

    if 'selected_file' in request.form:
        selected_file = request.form['selected_file']
        file_path = os.path.join(UPLOAD_FOLDER, selected_file)
        try:
            df = read_csv_file(file_path)
            if df is None:
                return "Error: The file format is not supported or the file is corrupted."
        except ValueError as e:
            return str(e)

        columns = df.columns.tolist()
        age_columns, sex_columns = find_columns(df)

        if not age_columns or not sex_columns:
            return "Error: The file must contain age and sex columns."

        age_column = age_columns[0]
        sex_column = sex_columns[0]
        age_range = get_column_range(df, age_column)
        sex_range = get_column_range(df, sex_column)

        context.update({
            'columns': columns,
            'age_column': age_column,
            'age_range': age_range,
            'sex_column': sex_column,
            'sex_range': sex_range,
            'file_path': file_path,
            'selected_file': selected_file
        })

    context['selected_file'] = selected_file

    # 读取referenzintervall.csv文件
    referenz_df = read_csv_file(REFERENZINTERVALL_FILE)
    if referenz_df is not None:
        referenz_df = referenz_df[referenz_df['ZEITRAUM'] == 'Jahr']
        unique_codes = referenz_df['CODE'].unique()  # 获取唯一的CODE
        context['referenz_data'] = referenz_df.to_dict(orient='records')
        context['unique_codes'] = unique_codes

    return render_template('index.html', **context)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        column = request.form['column']
        file_path = request.form['file_path']
        age_column = request.form['age_column']
        sex_column = request.form['sex_column']
        age_from = request.form['age_from']
        age_to = request.form['age_to']
        sex_value = request.form['sex_value']

        df = read_csv_file(file_path)
        if df is None:
            return jsonify({'error': "The file format is not supported or the file is corrupted."})

        df = df.dropna(subset=[age_column, sex_column, column])
        if df[age_column].dtype != 'int':
            df[age_column] = df[age_column].astype(float).astype(int)

        age_from = int(age_from)
        age_to = int(age_to)
        df = df[(df[age_column] >= age_from) & (df[age_column] <= age_to) & (
                    df[sex_column].str.lower() == sex_value.lower())]

        selected_column = df[column]

        # 清洗掉非数值型的值并转换为浮点数
        selected_column = pd.to_numeric(selected_column.str.replace(',', '.'), errors='coerce').dropna()

        digit_freq = leading_digit_analysis(selected_column)
        benford_freq = benford_distribution() * digit_freq.sum()  # 调整期望频率以匹配观察频率的总和
        total_count = len(selected_column.dropna())

        fig = go.Figure()
        fig.add_trace(go.Bar(x=digit_freq.index, y=digit_freq.values, name="Actual Data"))
        fig.add_trace(go.Scatter(x=list(range(1, 10)), y=benford_freq, mode='lines+markers', name="Benford's Law"))
        fig.update_layout(title=f"Leading Digit Distribution vs Benford's Law (Column: {column})",
                          xaxis_title='Leading Digit',
                          yaxis_title='Frequency',
                          xaxis=dict(tickmode='linear', dtick=1),
                          height=300)

        plot_html = pio.to_html(fig, full_html=False)

        return jsonify({
            'plot_html': plot_html,
            'digit_freq': digit_freq.tolist(),
            'benford_freq': benford_freq.tolist(),
            'total_count': total_count,
            'age_from': age_from,
            'age_to': age_to
        })
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})


@app.route('/stat_tests', methods=['POST'])
def stat_tests():
    try:
        data_source = request.form.get('data_source')
        digit_freq_str = request.form.get('digit_freq')
        benford_freq_str = request.form.get('benford_freq')
        total_count = int(request.form.get('total_count'))
        analysis_methods = request.form.getlist('analysis_methods')

        digit_freq = np.array([float(x) for x in digit_freq_str.split(',')])
        benford_freq = np.array([float(x) for x in benford_freq_str.split(',')])

        results = {}
        fig = go.Figure()

        if data_source == 'intervall_data':
            digit_freq = np.array([float(x) for x in request.form.get('digit_freq_all').split(',')])
            total_count = int(request.form.get('total_count_all'))
        elif data_source == 'normal_data':
            digit_freq = np.array([float(x) for x in request.form.get('digit_freq_normal').split(',')])
            total_count = int(request.form.get('total_count_normal'))

        if 'chi_square_test' in analysis_methods:
            chi2_stat = chi_square_test(digit_freq, benford_freq, total_count)
            results['chi_square_test'] = {'chi2_stat': chi2_stat}
            fig.add_trace(go.Scatter(x=np.arange(1, 10), y=benford_freq, mode='lines', name="Benford's Law"))
            fig.add_trace(go.Scatter(x=np.arange(1, 10), y=digit_freq, mode='lines', name="Actual Data"))
            if chi2_stat > 16.9:
                results['chi_square_test'][
                    'result'] = "Chi-Square value is greater than 16.9. The data does not conform to Benford's Law."
            else:
                results['chi_square_test'][
                    'result'] = "Chi-Square value is less than or equal to 16.9. The data conforms to Benford's Law."

        if 'z_test' in analysis_methods:
            z_scores, significant_digits = z_test(digit_freq, benford_freq, total_count)
            results['z_test'] = {'z_score': z_scores, 'significant_digits': significant_digits}
            fig.add_trace(go.Scatter(x=np.arange(1, 10), y=benford_freq, mode='lines', name="Benford's Law"))
            fig.add_trace(go.Scatter(x=np.arange(1, 10), y=digit_freq, mode='lines', name="Actual Data"))
            if significant_digits:
                results['z_test']['result'] = f"Digits {significant_digits} do not conform to Benford's Law."
            else:
                results['z_test']['result'] = "All digits conform to Benford's Law."

        if 'measure_of_distance' in analysis_methods:
            distance = measure_of_distance(digit_freq, benford_freq)
            results['measure_of_distance'] = {'distance': distance}
            fig.add_trace(go.Scatter(x=np.arange(1, 10), y=benford_freq, mode='lines', name="Benford's Law"))
            fig.add_trace(go.Scatter(x=np.arange(1, 10), y=digit_freq, mode='lines', name="Actual Data"))

        if 'regression_measure' in analysis_methods:
            slope, intercept = regression_measure(digit_freq)
            results['regression_measure'] = {'slope': slope, 'intercept': intercept}
            benford_proportions = np.log10(1 + 1 / np.arange(1, 10))
            observed_proportions = np.log10(digit_freq)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=benford_proportions, y=observed_proportions, mode='markers', name="Actual Data"))
            fig.add_trace(go.Scatter(x=benford_proportions, y=slope * benford_proportions + intercept, mode='lines',
                                     name='Regression Line'))
            fig.update_layout(title='Saville Regression Measure',
                              xaxis_title='Log10(Expected)',
                              yaxis_title='Log10(Observed)',
                              height=300)
            results['regression_plot_html'] = pio.to_html(fig, full_html=False)

        if fig.data:
            results['plot_html'] = pio.to_html(fig, full_html=False)

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})


@app.route('/get_intervals', methods=['POST'])
def get_intervals():
    try:
        code = request.form['code']
        age_from = int(request.form['age_from'])
        age_to = int(request.form['age_to'])
        sex_value = request.form['sex_value'].lower()
        referenz_df = read_csv_file(REFERENZINTERVALL_FILE)
        referenz_df = referenz_df[referenz_df['ZEITRAUM'] == 'Jahr']
        if referenz_df is not None:
            intervals = referenz_df[referenz_df['CODE'] == code]
            interval_matches = intervals.apply(
                lambda row: age_from >= row['ALTERVON'] and age_to <= row['ALTERBIS'] and (
                            row['SEX'].lower() == sex_value or row['SEX'].lower() == 'al'), axis=1)
            return jsonify(
                {'intervals': intervals.to_dict(orient='records'), 'interval_matches': interval_matches.tolist()})
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})


@app.route('/interval_analyze', methods=['POST'])
def interval_analyze():
    try:
        code = request.form['code']
        age_from = int(request.form['age_from'])
        age_to = int(request.form['age_to'])
        sex_value = request.form['sex_value'].lower()
        referenz_df = read_csv_file(REFERENZINTERVALL_FILE)
        referenz_df = referenz_df[referenz_df['ZEITRAUM'] == 'Jahr']
        file_path = request.form['file_path']
        column = request.form['column']

        df = read_csv_file(file_path)
        if df is None:
            return jsonify({'error': "The file format is not supported or the file is corrupted."})

        df = df.dropna(subset=[column])

        # 根据用户输入的年龄范围找到对应的 interval
        interval_filtered = referenz_df[(referenz_df['CODE'] == code) & (referenz_df['ALTERVON'] <= age_from) & (
                    referenz_df['ALTERBIS'] >= age_to) & ((referenz_df['SEX'].str.lower() == sex_value) | (
                    referenz_df['SEX'].str.lower() == 'al'))]
        if interval_filtered.empty:
            return jsonify({'error': "No matching interval found for the given age range and sex."})

        interval_age_from = interval_filtered['ALTERVON'].min()
        interval_age_to = interval_filtered['ALTERBIS'].max()

        # 过滤年龄范围和性别
        df_age_filtered = df[(df['ALTER'] >= interval_age_from) & (df['ALTER'] <= interval_age_to) & (
                    df['SEX'].str.lower() == sex_value)]

        # 清洗并转换列为浮点数
        df_age_filtered[column] = pd.to_numeric(df_age_filtered[column].str.replace(',', '.'), errors='coerce').dropna()

        # 获取 normal range 的数据
        normal_from = interval_filtered['NORMVON'].values[0]
        normal_to = interval_filtered['NORMBIS'].values[0]

        normal_from = float(normal_from.replace(',', '.')) if ',' in str(normal_from) else float(normal_from)
        normal_to = float(normal_to.replace(',', '.')) if ',' in str(normal_to) else float(normal_to)

        # 切片数据
        df_normal_filtered = df_age_filtered[
            (df_age_filtered[column] >= normal_from) & (df_age_filtered[column] <= normal_to)]

        # 分析年龄在 interval 范围内的所有数据
        digit_freq_all = leading_digit_analysis(df_age_filtered[column])
        benford_freq = benford_distribution() * digit_freq_all.sum()
        total_count_all = len(df_age_filtered[column].dropna())

        fig_all = go.Figure()
        fig_all.add_trace(go.Bar(x=digit_freq_all.index, y=digit_freq_all.values, name="Actual Data"))
        fig_all.add_trace(go.Scatter(x=list(range(1, 10)), y=benford_freq, mode='lines+markers', name="Benford's Law"))
        fig_all.update_layout(title=f"Leading Digit Distribution vs Benford's Law (All Data, Column: {column})",
                              xaxis_title='Leading Digit',
                              yaxis_title='Frequency',
                              xaxis=dict(tickmode='linear', dtick=1),
                              height=300)

        plot_html_all = pio.to_html(fig_all, full_html=False)

        # 分析 normal range 数据
        digit_freq_normal = leading_digit_analysis(df_normal_filtered[column])
        total_count_normal = len(df_normal_filtered[column].dropna())

        fig_normal = go.Figure()
        fig_normal.add_trace(go.Bar(x=digit_freq_normal.index, y=digit_freq_normal.values, name="Actual Data"))
        fig_normal.add_trace(
            go.Scatter(x=list(range(1, 10)), y=benford_freq, mode='lines+markers', name="Benford's Law"))
        fig_normal.update_layout(title=f"Leading Digit Distribution vs Benford's Law (Normal Range, Column: {column})",
                                 xaxis_title='Leading Digit',
                                 yaxis_title='Frequency',
                                 xaxis=dict(tickmode='linear', dtick=1),
                                 height=300)

        plot_html_normal = pio.to_html(fig_normal, full_html=False)

        return jsonify({
            'plot_html_all': plot_html_all,
            'plot_html_normal': plot_html_normal,
            'digit_freq_all': digit_freq_all.tolist(),
            'total_count_all': total_count_all,
            'digit_freq_normal': digit_freq_normal.tolist(),
            'total_count_normal': total_count_normal
        })
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5009)


#