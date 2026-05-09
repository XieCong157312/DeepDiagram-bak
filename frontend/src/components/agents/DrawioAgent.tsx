import { useEffect, useRef, useState, useImperativeHandle, forwardRef } from 'react';
import { useChatStore } from '../../store/chatStore';
import { cleanContent } from '../../lib/utils';
import type { AgentRef, AgentProps } from './types';

/**
 * Sanitize Draw.io XML by removing invalid <Array> elements.
 * The LLM sometimes generates <Array points="..."/> which Draw.io cannot parse.
 * Draw.io expects edge waypoints to be auto-routed, not manually specified.
 */
function sanitizeDrawioXml(xml: string): string {
    if (!xml || !xml.includes('<mxfile') && !xml.includes('<mxGraphModel')) {
        return xml;
    }
    // Remove <Array .../> self-closing elements
    let cleaned = xml.replace(/<Array[^/>]*\/>/g, '');
    // Remove <Array ...>...</Array> elements with content
    cleaned = cleaned.replace(/<Array[^>]*>[\s\S]*?<\/Array>/g, '');
    // Clean up empty mxGeometry elements that may result
    cleaned = cleaned.replace(/(<mxGeometry[^>]*>)\s*(<\/mxGeometry>)/g, '$1$2');
    // Clean up extra whitespace within mxGeometry
    cleaned = cleaned.replace(/(<mxGeometry[^>]*>)\s+(<\/(mxGeometry|mxCell)>)/g, '$1$2');
    return cleaned;
}

export const DrawioAgent = forwardRef<AgentRef, AgentProps>(({ content }, ref) => {
    const { isStreamingCode } = useChatStore();
    let currentCode = cleanContent(content);
    // Fix double-escaped newlines from LLM output
    if (currentCode.includes('\\n') && !currentCode.includes('\n')) {
        currentCode = currentCode.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
    }
    currentCode = currentCode.replace(/<br\/?>/gi, '&#xa;');
    // .replace(/<br\/?>/gi, "&lt;br/&gt;");

    console.log('DrawioAgent received content:', currentCode);
    const [iframeReady, setIframeReady] = useState(false);
    const drawioIframeRef = useRef<HTMLIFrameElement>(null);

    useImperativeHandle(ref, () => ({
        handleDownload: async (type: 'png' | 'svg') => {
            if (drawioIframeRef.current?.contentWindow) {
                drawioIframeRef.current.contentWindow.postMessage(JSON.stringify({
                    action: 'export',
                    format: type,
                    spin: true
                }), '*');
            } else {
                alert('Editor not ready.');
            }
        }
    }));

    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            if (!event.data || typeof event.data !== 'string') return;

            let msg;
            try {
                msg = JSON.parse(event.data);
            } catch (e) {
                return;
            }

            if (msg.event === 'export') {
                if (msg.data) {
                    const downloadFile = (url: string, ext: string) => {
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `deepdiagram-drawio-${new Date().getTime()}.${ext}`;
                        a.click();
                    };
                    downloadFile(msg.data, msg.format === 'xmlsvg' || msg.format === 'svg' ? 'svg' : 'png');
                }
            }
            if (msg.event === 'configure') {
                drawioIframeRef.current?.contentWindow?.postMessage(JSON.stringify({
                    action: 'configure',
                    config: {
                        compressXml: false,
                        sidebarVisible: false,
                        formatVisible: false,
                        showFormatPanel: false,
                        showStartScreen: false,
                        gridColor: '#f1f3f4',
                        // Ensure sidebars are hidden in all themes
                        sidebar_visible: false,
                        format_visible: false,
                        showSidebar: false,
                        libraries: false
                    }
                }), '*');
            }
            if (msg.event === 'init') {
                setIframeReady(true);
                // Force close panels after a delay to ensure the UI is fully loaded
                const win = drawioIframeRef.current?.contentWindow;
                setTimeout(() => {
                    win?.postMessage(JSON.stringify({ action: 'exec', cmd: 'formatPanel', value: false }), '*');
                    win?.postMessage(JSON.stringify({ action: 'exec', cmd: 'sidebar', value: false }), '*');
                }, 1000);
            }
            else if (msg.event === 'save' || msg.event === 'autosave') {
                if (msg.xml) {
                    // Drawio normally would want to save back, but here we prioritze LLM flow
                    // console.log('Drawio save event:', msg.xml);
                }
            }
        };

        window.addEventListener('message', handleMessage);
        return () => window.removeEventListener('message', handleMessage);
    }, []);

    const extractDrawioXml = (raw: string) => {
        let xml = raw.replace(/```xml\s?/, '').replace(/```/, '').trim();

        // Extract from XML-style tags if present
        const codeMatch = xml.match(/<code>\s*([\s\S]*?)\s*<\/code>/i);
        if (codeMatch) {
            xml = codeMatch[1].trim();
        }

        // Remove design_concept wrapper if it accidentally remains
        xml = xml.replace(/<design_concept>[\s\S]*?<\/design_concept>/gi, '').trim();

        return xml;
    };

    const escapeXmlEntities = (xml: string) => {
        return xml.replace(/&(?!(?:amp|lt|gt|quot|apos|#[0-9]+|#x[0-9A-Fa-f]+);)/g, '&amp;');
    };

    useEffect(() => {
        if (!isStreamingCode && iframeReady && currentCode && drawioIframeRef.current) {
            let cleanXml = extractDrawioXml(currentCode);

            // Handle nested JSON structure: {"design_concept": "...", "code": "..."}
            if (cleanXml.startsWith('{') && cleanXml.includes('"code"')) {
                try {
                    const parsed = JSON.parse(cleanXml);
                    if (parsed.code) {
                        cleanXml = typeof parsed.code === 'string' ? parsed.code : '';
                    }
                } catch {
                    // Not valid JSON, continue with original cleanXml
                }
            }

            if (cleanXml.startsWith('<')) {
                // Sanitize XML to remove invalid <Array> elements that cause parsing errors
                cleanXml = sanitizeDrawioXml(cleanXml);

                // Escape raw ampersands in XML content to prevent parser failures
                cleanXml = escapeXmlEntities(cleanXml);

                const win = drawioIframeRef.current.contentWindow;
                win?.postMessage(JSON.stringify({
                    action: 'load',
                    xml: cleanXml,
                    autosave: 1,
                    fit: 1
                }), '*');

                // Explicit fit after a short delay to ensure rendering is complete
                setTimeout(() => {
                    win?.postMessage(JSON.stringify({
                        action: 'fit',
                        padding: 20
                    }), '*');
                }, 1200);
            }
        }
    }, [iframeReady, currentCode, isStreamingCode]);

    // Use ui=atlas with explicit sidebar=0 and format=0 which are more reliable
    const drawioUrl = "https://embed.diagrams.net/?" + new URLSearchParams({
        embed: '1',
        ui: 'atlas',
        spin: '1',
        modified: 'unsavedChanges',
        proto: 'json',
        configure: '1',
        fit: '1',
        sidebar: '0',
        format: '0',
        libs: '0',
        menubar: '0',
        toolbar: '0',
        status: '0',
        noSaveBtn: '1',
        noExitBtn: '1'
    }).toString();

    return (
        <div className="w-full h-full">
            <iframe
                ref={drawioIframeRef}
                src={drawioUrl}
                className="w-full h-full border-none"
                title="Draw.io Editor"
            />
        </div>
    );
});
