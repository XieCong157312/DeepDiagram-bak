import { useEffect, useRef, useImperativeHandle, forwardRef, useState } from 'react';
import { useChatStore } from '../../store/chatStore';
import { cleanContent } from '../../lib/utils';
import * as echarts from 'echarts';
import type { AgentRef, AgentProps } from './types';
import { AlertCircle } from 'lucide-react';

export const ChartsAgent = forwardRef<AgentRef, AgentProps>(({ content }, ref) => {
    const { isStreamingCode } = useChatStore();
    let currentCode = cleanContent(content);
    console.log('ChartsAgent raw content:', currentCode);
    // Fix double-escaped newlines from LLM output
    if (currentCode.includes('\\n') && !currentCode.includes('\n')) {
        currentCode = currentCode.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
    }
    console.log('ChartsAgent received content:', currentCode);

    const chartRef = useRef<HTMLDivElement>(null);
    const chartInstanceRef = useRef<echarts.ECharts | null>(null);
    const [error, setError] = useState<string | null>(null);

    useImperativeHandle(ref, () => ({
        handleDownload: async (type: 'png' | 'svg') => {
            if (!chartInstanceRef.current) return;
            const filename = `deepdiagram - charts - ${new Date().getTime()} `;

            const downloadFile = (url: string, ext: string) => {
                const a = document.createElement('a');
                a.href = url;
                a.download = `${filename}.${ext} `;
                a.click();
            };

            if (type === 'png') {
                const url = chartInstanceRef.current.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
                downloadFile(url, 'png');
            } else {
                alert('SVG export for charts is not currently supported in this mode.');
            }
        }
    }));

    useEffect(() => {
        if (!currentCode || !chartRef.current) return;
        if (isStreamingCode) return;

        // Initialize or get existing instance
        let chart = chartInstanceRef.current;
        if (!chart) {
            chart = echarts.init(chartRef.current);
            chartInstanceRef.current = chart;
        }

        try {
            setError(null);
            let options: any;

            // Helper to try parsing JSON-like string
            const tryParse = (str: string) => {
                try {
                    return JSON.parse(str);
                } catch {
                    try {
                        return new Function(`return (${str})`)();
                    } catch {
                        return null;
                    }
                }
            };

            let cleanCode = currentCode.trim();
            options = tryParse(cleanCode);

            if (!options) {
                const match = cleanCode.match(/```(?:json|chart)?\s*([\s\S]*?)\s*```/i);
                if (match) {
                    options = tryParse(match[1].trim());
                }
            }

            if (!options) {
                const start = cleanCode.indexOf('{');
                const end = cleanCode.lastIndexOf('}');
                if (start !== -1 && end !== -1 && end > start) {
                    options = tryParse(cleanCode.substring(start, end + 1));
                }
            }

            if (!options) throw new Error("Could not parse chart configuration");

            // Remove rigid sizing so the chart can adapt to the container
            if (options && typeof options === 'object') {
                delete options.width;
                delete options.height;

                const normalizeGrid = (grid: any) => ({
                    left: grid?.left ?? '5%',
                    right: grid?.right ?? '5%',
                    top: grid?.top ?? '22%',
                    bottom: grid?.bottom ?? '12%',
                    containLabel: grid?.containLabel !== undefined ? grid.containLabel : true,
                    show: grid?.show !== undefined ? grid.show : true
                });

                if (Array.isArray(options.grid)) {
                    options.grid = options.grid.map(normalizeGrid);
                } else if (!options.grid || typeof options.grid !== 'object') {
                    options.grid = normalizeGrid({});
                } else {
                    options.grid = normalizeGrid(options.grid);
                }

                if (Array.isArray(options.grid) && Array.isArray(options.xAxis)) {
                    while (options.grid.length < options.xAxis.length) {
                        options.grid.push(normalizeGrid({ top: '60%', bottom: '12%' }));
                    }
                    if (options.grid.length > options.xAxis.length) {
                        options.grid = options.grid.slice(0, options.xAxis.length);
                    }
                }

                if (!options.tooltip) {
                    options.tooltip = { trigger: 'axis', confine: true };
                } else if (typeof options.tooltip === 'object') {
                    options.tooltip.confine = true;
                }

                if (!options.title) {
                    options.title = { left: 'center', top: 8, textStyle: { fontWeight: 700, fontSize: 18 }, subtextStyle: { fontSize: 12 } };
                } else if (typeof options.title === 'object') {
                    if (options.title.left == null) options.title.left = 'center';
                    if (options.title.top == null) options.title.top = 8;
                    if (!options.title.textStyle) options.title.textStyle = { fontWeight: 700, fontSize: 18 };
                    if (!options.title.subtextStyle) options.title.subtextStyle = { fontSize: 12 };
                }

                if (!options.legend || typeof options.legend !== 'object') {
                    options.legend = { top: '15%', left: 'center', type: 'scroll', orient: 'horizontal', itemGap: 16 };
                } else {
                    if (options.legend.top == null) options.legend.top = '15%';
                    if (options.legend.left == null) options.legend.left = 'center';
                    if (options.legend.type == null) options.legend.type = 'scroll';
                    if (options.legend.orient == null) options.legend.orient = 'horizontal';
                    if (options.legend.itemGap == null) options.legend.itemGap = 16;
                }

                if (!options.backgroundColor) {
                    options.backgroundColor = 'transparent';
                }

                const ensureGridTop = (grid: any) => {
                    const topVal = parseInt(String(grid.top).replace('%', ''), 10);
                    if (!Number.isNaN(topVal) && topVal < 20) {
                        grid.top = '20%';
                    }
                    if (String(grid.bottom).endsWith('%')) {
                        const bottomVal = parseInt(String(grid.bottom).replace('%', ''), 10);
                        if (!Number.isNaN(bottomVal) && bottomVal < 10) {
                            grid.bottom = '12%';
                        }
                    }
                };

                if (options.title && (options.title.text || options.title.subtext)) {
                    if (Array.isArray(options.grid)) {
                        options.grid.forEach((grid: any) => ensureGridTop(grid));
                    } else {
                        ensureGridTop(options.grid);
                    }
                }
            }

            // Handle nested structure: {"design_concept": "...", "code": "..."}
            if (options.code && !options.series && !options.xAxis && !options.yAxis) {
                if (typeof options.code === 'string') {
                    options = tryParse(options.code);
                } else {
                    options = options.code;
                }
                if (!options) throw new Error("Could not parse chart configuration from code field");
            }

            // Auto-enrichment for consistency
            const hasXAxis = Array.isArray(options.xAxis) ? options.xAxis.length > 0 : !!options.xAxis;
            const hasYAxis = Array.isArray(options.yAxis) ? options.yAxis.length > 0 : !!options.yAxis;

            if (hasXAxis && hasYAxis) {
                if (!options.dataZoom && !options.series?.some((s: any) => s.type === 'pie')) {
                    options.dataZoom = [
                        { type: 'inside', xAxisIndex: [0], filterMode: 'filter' },
                        { type: 'slider', xAxisIndex: [0], filterMode: 'filter' }
                    ];
                }
                if (!options.tooltip) options.tooltip = { trigger: 'axis', confine: true };
            }

            if (Array.isArray(options.xAxis) && Array.isArray(options.grid) && options.grid.length === 1) {
                options.grid = options.grid.concat(options.grid.map((g: any) => ({ ...g, top: '60%', bottom: '12%' })).slice(0, options.xAxis.length - 1));
            }

            if (Array.isArray(options.grid) && Array.isArray(options.xAxis) && options.grid.length > 1) {
                options.grid = options.grid.map((grid: any, index: number) => ({
                    ...grid,
                    top: grid.top || `${20 + index * 40}%`,
                    bottom: grid.bottom || '12%',
                    containLabel: true
                }));
            }

            if (options.series) {
                options.series = options.series.map((s: any) => {
                    const type = s.type;
                    if (['graph', 'tree', 'map', 'sankey'].includes(type)) {
                        return { roam: true, ...s };
                    }
                    return s;
                });
            }

            // use notMerge: true to prevent leakage of old axis/grid state
            chart.setOption(options, { notMerge: true });

            useChatStore.getState().reportSuccess();

            const resizeObserver = new ResizeObserver(() => {
                try {
                    chart?.resize();
                } catch (e) {
                    console.warn("Chart resize error:", e);
                }
            });
            resizeObserver.observe(chartRef.current);

            return () => {
                resizeObserver.disconnect();
            };
        } catch (e) {
            console.error("ECharts error", e);
            const msg = e instanceof Error ? e.message : "Failed to render chart";
            setError(msg);
            if (chartInstanceRef.current) {
                chartInstanceRef.current.dispose();
                chartInstanceRef.current = null;
            }
        }
    }, [currentCode, isStreamingCode]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (chartInstanceRef.current) {
                chartInstanceRef.current.dispose();
                chartInstanceRef.current = null;
            }
        };
    }, []);

    return (
        <div className="absolute inset-0 w-full h-full bg-white flex flex-col items-center justify-center">
            {error ? (
                <div key="error-view" className="flex flex-col items-center justify-center p-8 text-center max-w-md">
                    <div className="p-4 bg-red-50 rounded-full mb-4">
                        <AlertCircle className="w-8 h-8 text-red-500" />
                    </div>
                    <p className="text-base font-semibold text-slate-800 mb-2">Chart Render Failed</p>
                    <p className="text-sm text-slate-600 mb-6">{error}</p>
                    <button
                        onClick={() => window.dispatchEvent(new CustomEvent('deepdiagram-retry', {
                            detail: {
                                index: useChatStore.getState().messages.length - 1,
                                error: error
                            }
                        }))}
                        className="px-6 py-2.5 bg-slate-900 text-white rounded-lg text-sm font-medium hover:bg-slate-800 transition-colors shadow-sm"
                    >
                        Try Regenerating
                    </button>
                </div>
            ) : (
                <div key="chart-view" ref={chartRef} className="w-full h-full" />
            )}
        </div>
    );
});
